"""
Stage 2.1 additional analyses: calibration and decision-curve analysis (DCA).

See EXPERIMENT_PLAN_DETAILS.md Stage 2.1 "additional analyses" section for the
literature this is based on (KFRE external-validation studies, CKD deep-learning
papers). Feature importance (feature_importance_analysis.py) says which inputs a
model leans on; this module asks two different questions per model/scenario:

1. Calibration — do predicted risks match observed outcomes? (per-decile
   predicted-vs-KM-observed table, plus Brier score, at 2-year/5-year horizons)
2. Decision curve analysis — would using this model's risk score to guide a
   referral decision do more good than harm, compared to treating everyone,
   treating no one, or an eGFR-threshold rule (the KFRE papers' own comparator)?

Competing-risk analysis (death before ESRD) is deliberately NOT implemented
here: the exported <scenario>_train/test_data.csv files only carry
`duration_in_days` (days since each subject's first lab record), not each
subject's absolute anchor timestamp, so `patients.csv`'s `dod` can't be
converted to "days since anchor" without re-deriving that anchor from the raw
extraction — the exact kind of expensive, easy-to-trigger-by-accident step
CLAUDE.md's "check a script's entry point" rule warns about. Raised with the
user and explicitly declined (2026-08-23) rather than worth the extraction
pipeline change — not planned, see EXPERIMENT_PLAN_DETAILS.md Stage 2.1.

Design choice — one prediction pipeline per model, one shared risk->survival
conversion for all of them: each model's "risk score" per test-set row is
extracted using that architecture's OWN already-validated prediction code path
(same Dataset classes, same forward pass used by that model's C-index/Brier/AUC
evaluation in pkgs/experiments/*.py), not re-derived from scratch. All 5 models'
risk scores are then converted to predicted survival probabilities with the
SAME exponential approximation S(t) = exp(-risk_norm * t/365) already used by
pkgs/experiments/utils.py's compute_brier_score_from_risk_scores — applied
uniformly so calibration/DCA numbers are comparable model-to-model within a
scenario, at the cost of not using e.g. logistic_hazard's own real predicted
survival curve (pycox's model.predict_surv_df). This mirrors an approximation
the codebase already relies on elsewhere, rather than introducing a new one.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from lifelines import KaplanMeierFitter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pkgs.commons import (
    current_rep, generate_data_path_latest_rep,
    four_features_train_data_path, four_features_test_data_path,
    eight_features_train_data_path, eight_features_test_data_path,
    twenty_features_heterogeneous_train_data_path, twenty_features_heterogeneous_test_data_path,
)
from pkgs.data_analysis.types import ExperimentScenario
from pkgs.experiments.utils import (
    load_pkl_and_dill_model, get_tv_rnn_model_features,
    compute_brier_score_from_survival_probs,
)
from pkgs.experiments.hazard_transformer import HazardTransformerDataset, custom_collate_fn
from pkgs.experiments.logistic_hazard import LogisticHazardDataset
from pkgs.experiments.dynamic_deephit import DynamicDeepHitDataset
from torch.utils.data import DataLoader
from pycox.models import LogisticHazard
import torch.optim as optim

# 2-year / 5-year, the horizons KFRE validation studies (Tangri et al. and its
# external validations — see EXPERIMENT_PLAN_DETAILS.md Stage 2.1 sources) report.
DEFAULT_HORIZONS_DAYS = [730, 1825]
# eGFR referral threshold used as the non-model comparator in DCA, per KDIGO/
# nephrology-referral convention and the KFRE decision-curve papers reviewed.
EGFR_REFERRAL_CUTOFFS = [30, 45]
DCA_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def risk_scores_to_survival_probs(risk_scores, times):
    """Same transform as pkgs/experiments/utils.py's compute_brier_score_from_risk_scores,
    factored out so callers can get the (n_samples, n_times) survival-probability
    matrix itself, not just the final integrated Brier score."""
    risk_scores = np.asarray(risk_scores, dtype=np.float64)
    risk_scores_norm = risk_scores - risk_scores.min() + 0.01
    times = np.asarray(times, dtype=np.float64)
    survival_probs = np.exp(-np.outer(risk_scores_norm, times) / 365.0)
    return survival_probs


def predicted_event_prob_at(risk_scores, horizon_days):
    """1 - S(horizon) per row, from the shared risk-score->survival-probability transform."""
    surv = risk_scores_to_survival_probs(risk_scores, [horizon_days])[:, 0]
    return 1.0 - surv


def _km_event_prob_at(durations, events, horizon_days):
    """Observed P(event by horizon) via Kaplan-Meier (handles censoring), or
    None if there isn't enough data in this group to fit one."""
    durations = np.asarray(durations, dtype=np.float64)
    events = np.asarray(events, dtype=np.float64)
    if len(durations) < 3:
        return None
    try:
        kmf = KaplanMeierFitter()
        kmf.fit(durations, event_observed=events)
        surv_at_horizon = kmf.survival_function_at_times(horizon_days)
        return float(1.0 - surv_at_horizon.values[0])
    except Exception:
        return None


def calibration_table(risk_scores, durations, events, horizon_days, n_bins=10):
    """Per-decile predicted-risk-vs-KM-observed-risk table at one horizon."""
    predicted = predicted_event_prob_at(risk_scores, horizon_days)
    df = pd.DataFrame({'predicted': predicted, 'duration': durations, 'event': events})

    try:
        df['bin'] = pd.qcut(df['predicted'], q=n_bins, duplicates='drop')
    except ValueError:
        # Too few distinct predicted values for n_bins quantile cuts (small
        # rep99 samples, or a model whose risk scores collapsed to near-ties).
        df['bin'] = pd.qcut(df['predicted'].rank(method='first'), q=min(n_bins, len(df)),
                             duplicates='drop')

    rows = []
    for bin_label, group in df.groupby('bin', observed=True):
        observed = _km_event_prob_at(group['duration'], group['event'], horizon_days)
        rows.append({
            'n': len(group),
            'events': int(group['event'].sum()),
            'mean_predicted_risk': round(float(group['predicted'].mean()), 4),
            'km_observed_risk': round(observed, 4) if observed is not None else None,
        })
    return rows


def treat_all_net_benefit_curve(durations, events, horizon_days, thresholds=DCA_THRESHOLDS):
    """Net benefit of "treat everyone" at each threshold — computed once from the
    full test set's own outcomes, independent of any model. Previously this was
    computed per-model instead (redundantly, and inconsistently for
    hazard_transformer/ddh, whose predictions cover a different row set than
    df_test) — moved out so there's one canonical curve every model is compared
    against."""
    overall_event_prob = _km_event_prob_at(durations, events, horizon_days)
    result = {}
    for pt in thresholds:
        if overall_event_prob is None:
            result[pt] = None
        else:
            result[pt] = round(
                overall_event_prob - (1 - overall_event_prob) * (pt / (1 - pt)), 5)
    return result


def model_net_benefit_curve(risk_scores, durations, events, horizon_days, thresholds=DCA_THRESHOLDS):
    """Net benefit of the model's risk score at each threshold, per Vickers &
    Elkin's decision-curve-analysis formula, with P(event by horizon | risk>=pt)
    estimated via Kaplan-Meier so censoring before `horizon_days` doesn't bias
    the count (a plain proportion would)."""
    predicted = predicted_event_prob_at(risk_scores, horizon_days)
    n = len(predicted)

    model_nb = {}
    for pt in thresholds:
        high_risk = predicted >= pt
        n_high = int(high_risk.sum())

        if n_high == 0:
            model_nb[pt] = 0.0
        else:
            p_event_given_high = _km_event_prob_at(
                np.asarray(durations)[high_risk], np.asarray(events)[high_risk], horizon_days)
            if p_event_given_high is None:
                model_nb[pt] = None
            else:
                tp_rate = p_event_given_high * (n_high / n)
                fp_rate = (1 - p_event_given_high) * (n_high / n)
                model_nb[pt] = round(tp_rate - fp_rate * (pt / (1 - pt)), 5)

    return model_nb


def egfr_threshold_net_benefit(egfr_values, durations, events, horizon_days, egfr_cutoffs=EGFR_REFERRAL_CUTOFFS):
    """Net benefit of a fixed eGFR<cutoff referral rule (the non-model comparator
    KFRE's own clinical-utility papers use), at the same horizon as the model curve."""
    egfr_values = np.asarray(egfr_values, dtype=np.float64)
    n = len(egfr_values)
    results = {}
    for cutoff in egfr_cutoffs:
        high_risk = egfr_values < cutoff
        n_high = int(high_risk.sum())
        if n_high == 0:
            results[cutoff] = 0.0
            continue
        p_event_given_high = _km_event_prob_at(
            np.asarray(durations)[high_risk], np.asarray(events)[high_risk], horizon_days)
        if p_event_given_high is None:
            results[cutoff] = None
            continue
        tp_rate = p_event_given_high * (n_high / n)
        fp_rate = (1 - p_event_given_high) * (n_high / n)
        # eGFR-threshold rule has no single "probability threshold" pt of its own;
        # KFRE clinical-utility papers report its net benefit as one point per
        # cutoff, for the reader to compare against the model's NB curve across pt.
        # Use n_high/n as a stand-in "implied threshold" only to keep the formula's
        # shape consistent — the eGFR rule itself doesn't vary by pt.
        implied_pt = n_high / n if n_high < n else 0.5
        results[cutoff] = round(tp_rate - fp_rate * (implied_pt / max(1 - implied_pt, 1e-6)), 5)
    return results


def brier_score_up_to(df_train, df_test, risk_scores, horizon_days):
    times = np.linspace(1, horizon_days, 50)
    survival_probs = risk_scores_to_survival_probs(risk_scores, times)
    return compute_brier_score_from_survival_probs(df_train, df_test, survival_probs, times)


# --- per-model risk-score extraction, reusing each architecture's own eval code path ---

def cox_predictions(model, df_test):
    risk_scores = model.predict_partial_hazard(df_test).values.flatten()
    return risk_scores, df_test['duration_in_days'].values, df_test['has_esrd'].values


def hazard_transformer_predictions(model, df_test, scenario):
    dataloader = DataLoader(HazardTransformerDataset(df_test, scenario), shuffle=False,
                             collate_fn=custom_collate_fn, batch_size=256)
    all_risk_scores, all_times, all_events = [], [], []
    model.eval()
    with torch.no_grad():
        for features, mask, time_to_events, event_indicators, _, _ in dataloader:
            hazard_preds, _, _ = model(features, mask)
            surv = torch.cumprod(1 - hazard_preds[:, 0, :], dim=1)
            risk_scores = (1 - surv[:, -1]).detach().cpu().numpy()
            all_risk_scores.extend(risk_scores)
            all_times.extend(time_to_events.squeeze().numpy())
            all_events.extend(event_indicators.squeeze().numpy())
    return np.array(all_risk_scores), np.array(all_times), np.array(all_events)


def logistic_hazard_predictions(net, df_test, scenario):
    # The .pt file saves the raw MLPVanilla net (see logistic_hazard.py's run()),
    # not the pycox LogisticHazard wrapper that has predict_surv_df — re-wrap it
    # the same way run() does before loading it here.
    model = LogisticHazard(net, optimizer=optim.Adam(net.parameters()))
    test_dataset = LogisticHazardDataset(df_test, scenario)
    x_test, durations_test, events_test = test_dataset.prepare_data_for_pycox()
    x_test = torch.tensor(x_test, dtype=torch.float32)
    surv = model.predict_surv_df(x_test)
    median_time_idx = np.argmin(np.abs(surv.index - np.median(durations_test)))
    risk_scores = 1 - surv.iloc[median_time_idx].values
    return risk_scores, durations_test, events_test


def dynamic_deephit_predictions(model, df_test, scenario):
    dataloader = DataLoader(DynamicDeepHitDataset(df_test, scenario), shuffle=False, batch_size=16)
    all_risk_scores, all_times, all_events = [], [], []
    model.eval()
    with torch.no_grad():
        for features, mask, time_to_event, event_indicator, time_to_events, event_indicators, seq_lens in dataloader:
            hazard_preds, _ = model(features, mask)
            hazard_preds = hazard_preds.cpu().detach().numpy()[:, 0, :]
            for j in range(hazard_preds.shape[0]):
                p_seq_len = int(seq_lens[j])
                all_risk_scores.extend(hazard_preds[j][:p_seq_len])
                all_times.extend(time_to_events[j][:p_seq_len].cpu().detach().numpy())
                all_events.extend(event_indicators[j][:p_seq_len].cpu().detach().numpy())
    return np.array(all_risk_scores), np.array(all_times), np.array(all_events)


def rnn_surv_predictions(model, df_test, scenario):
    features = get_tv_rnn_model_features(scenario)
    X_test = torch.tensor(df_test[features].values, dtype=torch.float32).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        _, test_risk_scores = model(X_test)
        test_risk_scores = test_risk_scores.squeeze()
    risk_scores = 1 - test_risk_scores.cpu().numpy()
    return risk_scores, df_test['duration_in_days'].values, df_test['has_esrd'].values


class ClinicalValidityAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.current_scenario = None
        self.scenario_report_lines = {}
        self.models = ['cox', 'ddh', 'hazard_transformer', 'logistic_hazard', 'rnn_surv']
        self.model_pretty_names = {
            'cox': 'Cox', 'ddh': 'Dynamic DeepHit', 'hazard_transformer': 'Hazard Transformer',
            'logistic_hazard': 'Logistic Hazard', 'rnn_surv': 'RNN-Surv',
        }
        # scenario_name -> horizon_days -> {'treat_all':.., 'egfr_nb':.., 'models': {model_name: {'calibration':.., 'model_nb':..}}}
        # populated by analyze_scenario, read back by create_calibration_plot/create_decision_curve_plot.
        self.all_results = {}

    def log(self, message):
        print(message)
        if self.current_scenario is not None:
            self.scenario_report_lines.setdefault(self.current_scenario, []).append(message)

    def _model_paths(self, scenario_name):
        return {
            'cox': generate_data_path_latest_rep + f'/{scenario_name}_cox_model.dill',
            'ddh': generate_data_path_latest_rep + f'/{scenario_name}_ddh_model.pt',
            'hazard_transformer': generate_data_path_latest_rep + f'/{scenario_name}_hazard_transformer_model.pt',
            'logistic_hazard': generate_data_path_latest_rep + f'/{scenario_name}_logistic_hazard_model.pt',
            'rnn_surv': generate_data_path_latest_rep + f'/{scenario_name}_rnn_surv_model.pt',
        }

    def _get_predictions(self, model_name, model_path, df_test, scenario):
        if model_name == 'cox':
            model = load_pkl_and_dill_model(model_path)
            if model is None:
                return None
            return cox_predictions(model, df_test)

        model = torch.load(model_path, map_location='cpu', weights_only=False)
        if model_name == 'hazard_transformer':
            return hazard_transformer_predictions(model, df_test, scenario)
        if model_name == 'logistic_hazard':
            return logistic_hazard_predictions(model, df_test, scenario)
        if model_name == 'ddh':
            return dynamic_deephit_predictions(model, df_test, scenario)
        if model_name == 'rnn_surv':
            return rnn_surv_predictions(model, df_test, scenario)
        raise ValueError(f"Unknown model_name {model_name}")

    def analyze_scenario(self, scenario_name, scenario_enum, df_train, df_test):
        self.current_scenario = scenario_name
        self.log("=" * 80)
        self.log(f"CLINICAL VALIDITY ANALYSIS - {scenario_name.upper()}")
        self.log("=" * 80)
        self.log(f"Test samples (rows): {len(df_test)}")

        max_followup = df_test['duration_in_days'].max()
        horizons = [h for h in DEFAULT_HORIZONS_DAYS if h < max_followup]
        if not horizons:
            horizons = [max_followup * 0.5]
        self.log(f"Max test follow-up: {max_followup:.1f} days. Horizons used: {horizons}")
        self.log("")

        has_egfr = 'egfr' in df_test.columns

        # Get every model's predictions once up front — risk_scores/durations/
        # events don't depend on horizon, only the model does.
        predictions = {}
        for model_name, model_path in self._model_paths(scenario_name).items():
            if not os.path.exists(model_path):
                self.log(f"Model file not found, skipping: {model_path}")
                continue
            try:
                predictions[model_name] = self._get_predictions(
                    model_name, model_path, df_test, scenario_enum)
            except Exception as e:
                self.log(f"Error getting predictions for {model_name}: {e}")

        scenario_results = {}

        for horizon in horizons:
            self.log(f"\n=== Horizon: {horizon:.0f} days ===")

            # Computed once per horizon, from the full df_test, independent of
            # any model — every model's panel below is compared against these.
            treat_all = treat_all_net_benefit_curve(
                df_test['duration_in_days'].values, df_test['has_esrd'].values, horizon)
            self.log(f"Treat-all net benefit: {treat_all}")

            egfr_nb = None
            if has_egfr:
                try:
                    egfr_nb = egfr_threshold_net_benefit(
                        df_test['egfr'].values, df_test['duration_in_days'].values,
                        df_test['has_esrd'].values, horizon)
                    self.log(f"eGFR-threshold referral rule net benefit: {egfr_nb}")
                except Exception as e:
                    self.log(f"Error computing eGFR-threshold net benefit: {e}")

            horizon_result = {'treat_all': treat_all, 'egfr_nb': egfr_nb, 'models': {}}

            for model_name, preds in predictions.items():
                risk_scores, durations, events = preds
                self.log(f"\n--- {self.model_pretty_names[model_name]} ---")

                table = None
                self.log("Calibration (predicted risk decile vs. KM-observed risk):")
                try:
                    table = calibration_table(risk_scores, durations, events, horizon)
                    for row in table:
                        self.log(f"  n={row['n']:>4} events={row['events']:>3} "
                                  f"predicted={row['mean_predicted_risk']:.4f} "
                                  f"observed(KM)={row['km_observed_risk']}")
                except Exception as e:
                    self.log(f"  Error computing calibration table: {e}")

                try:
                    brier = brier_score_up_to(df_train, df_test, risk_scores, horizon)
                    self.log(f"Integrated Brier score (0-{horizon:.0f}d): {brier}")
                except Exception as e:
                    self.log(f"  Error computing Brier score: {e}")

                model_nb = None
                self.log("Decision curve analysis (net benefit by risk threshold):")
                try:
                    model_nb = model_net_benefit_curve(risk_scores, durations, events, horizon)
                    for pt in DCA_THRESHOLDS:
                        self.log(f"  pt={pt:.2f}  model={model_nb[pt]}  "
                                  f"treat_all={treat_all[pt]}  treat_none=0.0")
                except Exception as e:
                    self.log(f"  Error computing decision curve: {e}")

                horizon_result['models'][model_name] = {'calibration': table, 'model_nb': model_nb}

            scenario_results[horizon] = horizon_result

        self.all_results[scenario_name] = scenario_results
        self.create_calibration_plot(scenario_name, horizons)
        self.create_decision_curve_plot(scenario_name, horizons)
        self.save_scenario_report(scenario_name)

    def create_calibration_plot(self, scenario_name, horizons):
        """<scenario>_calibration_plot.png — rows=horizons, cols=models. Each
        panel: mean predicted risk per decile (x) vs. KM-observed risk per
        decile (y), with a dashed y=x reference diagonal."""
        try:
            results = self.all_results[scenario_name]
            models_with_data = [
                m for m in self.models
                if any(results[h]['models'].get(m, {}).get('calibration') for h in horizons)
            ]
            if not models_with_data:
                self.log("No calibration data available to plot.")
                return

            n_rows, n_cols = len(horizons), len(models_with_data)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

            for i, horizon in enumerate(horizons):
                for j, model_name in enumerate(models_with_data):
                    ax = axes[i][j]
                    table = results[horizon]['models'].get(model_name, {}).get('calibration')

                    if not table:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        xs_all = [row['mean_predicted_risk'] for row in table]
                        xy = [(row['mean_predicted_risk'], row['km_observed_risk'])
                              for row in table if row['km_observed_risk'] is not None]
                        if xy:
                            xs, ys = zip(*xy)
                            ax.plot(xs, ys, 'o-', color='tab:blue', markersize=4)
                        upper = max([1.0] + xs_all)
                        ax.plot([0, upper], [0, upper], '--', color='gray', linewidth=1)
                        ax.set_xlim(0, upper)
                        ax.set_ylim(0, upper)

                    if i == 0:
                        ax.set_title(self.model_pretty_names[model_name], fontsize=10)
                    if j == 0:
                        ax.set_ylabel(f'{horizon:.0f}d horizon\nObserved (KM)', fontsize=9)
                    if i == n_rows - 1:
                        ax.set_xlabel('Predicted', fontsize=9)

            plt.tight_layout()
            output_path = self.output_dir / f'{scenario_name}_calibration_plot.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            self.log(f"Calibration plot saved to: {output_path}")
        except Exception as e:
            self.log(f"Error creating calibration plot: {e}")

    def create_decision_curve_plot(self, scenario_name, horizons):
        """<scenario>_decision_curve_plot.png — one subplot per horizon: net
        benefit (y) vs. risk threshold (x), one line per model plus treat-all/
        treat-none reference lines and eGFR-cutoff reference points."""
        try:
            results = self.all_results[scenario_name]
            fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 5), squeeze=False)
            axes = axes[0]
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.models)))

            for i, horizon in enumerate(horizons):
                ax = axes[i]
                horizon_result = results[horizon]

                for color, model_name in zip(colors, self.models):
                    model_nb = horizon_result['models'].get(model_name, {}).get('model_nb')
                    if not model_nb:
                        continue
                    xy = [(pt, model_nb[pt]) for pt in DCA_THRESHOLDS if model_nb.get(pt) is not None]
                    if xy:
                        xs, ys = zip(*xy)
                        ax.plot(xs, ys, '-o', label=self.model_pretty_names[model_name],
                                color=color, markersize=3)

                treat_all = horizon_result['treat_all']
                xy = [(pt, treat_all[pt]) for pt in DCA_THRESHOLDS if treat_all.get(pt) is not None]
                if xy:
                    xs, ys = zip(*xy)
                    ax.plot(xs, ys, '--', color='black', label='Treat all')
                ax.axhline(0.0, linestyle=':', color='gray', label='Treat none')

                egfr_nb = horizon_result.get('egfr_nb')
                if egfr_nb:
                    for cutoff, nb in egfr_nb.items():
                        if nb is not None:
                            ax.scatter([0.1], [nb], marker='D', color='red', zorder=5)
                            ax.annotate(f'eGFR<{cutoff}', (0.1, nb), fontsize=7, color='red',
                                        xytext=(5, 0), textcoords='offset points')

                ax.set_title(f'{horizon:.0f}-day horizon')
                ax.set_xlabel('Risk threshold (pt)')
                if i == 0:
                    ax.set_ylabel('Net benefit')
                ax.legend(fontsize=7)

            plt.tight_layout()
            output_path = self.output_dir / f'{scenario_name}_decision_curve_plot.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            self.log(f"Decision curve plot saved to: {output_path}")
        except Exception as e:
            self.log(f"Error creating decision curve plot: {e}")

    def save_scenario_report(self, scenario_name):
        lines = self.scenario_report_lines.get(scenario_name, [])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = [
            "CLINICAL VALIDITY ANALYSIS REPORT (calibration + decision-curve analysis)",
            "=" * 80,
            f"Generated on: {timestamp}",
            f"Repetition: {current_rep}",
            f"Scenario: {scenario_name}",
            "Competing-risk analysis (death before ESRD) not included — raised and declined",
            "(2026-08-23); see EXPERIMENT_PLAN_DETAILS.md Stage 2.1 for why.",
            "=" * 80,
            "",
        ]
        report_path = self.output_dir / f'{scenario_name}_clinical_validity_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(header + lines))
        print(f"Scenario report saved to: {report_path}")
        self.current_scenario = None

    def analyze_four_features(self):
        from pkgs.data_analysis.model_data_store import get_train_test_data
        df_train, df_test = get_train_test_data(ExperimentScenario.FOUR_FEATURES)
        self.analyze_scenario('four_features', ExperimentScenario.FOUR_FEATURES, df_train, df_test)

    def analyze_eight_features(self):
        from pkgs.data_analysis.model_data_store import get_train_test_data
        df_train, df_test = get_train_test_data(ExperimentScenario.EIGHT_FEATURES)
        self.analyze_scenario('eight_features', ExperimentScenario.EIGHT_FEATURES, df_train, df_test)

    def analyze_twenty_features(self):
        from pkgs.data_analysis.model_data_store import get_train_test_data
        df_train, df_test = get_train_test_data(ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS)
        self.analyze_scenario('twenty_features_heterogeneous', ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS,
                               df_train, df_test)
