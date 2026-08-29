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

Design choice — one prediction pipeline per model, one calibrated risk->survival
conversion PER MODEL: each model's "risk score" per test-set row is extracted
using that architecture's OWN already-validated prediction code path (same
Dataset classes, same forward pass used by that model's C-index/Brier/AUC
evaluation in pkgs/experiments/*.py), not re-derived from scratch. Whenever a
model has no native per-horizon output (or the requested horizon falls outside
what it natively covers), its risk score is converted to a predicted survival
probability via a per-model Breslow-style baseline cumulative hazard —
fit_breslow_baseline_hazard() — fit from that SAME model's own risk scores on
the TRAINING set, then combined as S(t) = exp(-r_shifted * H0(t))
(calibrated_survival_probs()).

An earlier version used one hardcoded formula, S(t) = exp(-risk_norm * t/365)
(still what pkgs/experiments/utils.py's compute_brier_score_from_risk_scores
uses, for the older scenarios' own training/eval scripts — deliberately not
touched here, out of scope), applied identically to every model regardless of
its risk score's actual scale. That produced Stage 2.2 Finding #2 (see
generated_data/rep99/stage2_2_debug_report.txt): different architectures' raw
risk scores span wildly different magnitudes (a Cox partial hazard vs. an SVM
decision score vs. a GBSA score), so a fixed conversion collapsed to ~constant
near-1.0 predicted risk for 6 of 11 models, every scenario — visible as the
repeated "predicted risk is the same for every patient" report warnings.
Fitting the baseline per model, from that model's own training data, removes
the fixed-scale assumption entirely rather than picking a different fixed
constant — each model's own risk-score distribution determines its baseline,
so this stays comparable model-to-model (same C-index-style "higher=riskier"
convention, same fitting procedure) without assuming they share one absolute
scale.
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
from lifelines.utils import concordance_index
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pkgs.commons import (
    current_rep, generate_data_path_latest_rep,
    four_features_train_data_path, four_features_test_data_path,
    eight_features_train_data_path, eight_features_test_data_path,
    twenty_features_heterogeneous_train_data_path, twenty_features_heterogeneous_test_data_path,
)
from pkgs.data_analysis.types import ExperimentScenario
from pkgs.experiments.utils import load_pkl_and_dill_model
from pkgs.experiments.kfre import get_kfre_risk_scores_path

# Every model's own "how to turn raw output into (risk_scores, durations,
# events, native_prob_fn)" logic -- including fetching its own train/test
# data and building its own Dataset/DataLoader/tensor -- is a
# predictions(scenario, split='test') method on that model's own class in
# pkgs/models/ (deepsurv/dynamicdeephit/hazard_transformer/rnnsurv already
# had an architecture class there; cox/kfre/logistic_hazard/gbsa/srf/
# survival_svm/weibul had none, since they're direct calls into
# lifelines/pycox/sksurv, so a thin wrapper class holding the fitted
# estimator was added). This module only loads the file and dispatches to
# the right class -- see ClinicalValidityAnalyzer._get_predictions.
from pkgs.models.cox import CoxModel
from pkgs.models.logistic_hazard import LogisticHazardModel
from pkgs.models.gbsa import GBSAModel
from pkgs.models.srf import SRFModel
from pkgs.models.survival_svm import SurvivalSVMModel
from pkgs.models.weibul import WeibulModel
from pkgs.models.kfre import KFREModel

# 2-year / 5-year, the horizons KFRE validation studies (Tangri et al. and its
# external validations — see EXPERIMENT_PLAN_DETAILS.md Stage 2.1 sources) report.
DEFAULT_HORIZONS_DAYS = [730, 1825]
# eGFR referral threshold used as the non-model comparator in DCA, per KDIGO/
# nephrology-referral convention and the KFRE decision-curve papers reviewed.
EGFR_REFERRAL_CUTOFFS = [30, 45]
DCA_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def fit_breslow_baseline_hazard(train_risk_scores, train_durations, train_events):
    """Non-parametric (Breslow) baseline cumulative hazard H0(t), fit from a
    model's OWN risk scores on the TRAINING set, under the same
    proportional-hazards convention already used everywhere else in this
    module ("higher risk_score = proportionally higher hazard" -- the same
    assumption that lets C-index treat risk_scores as comparable rankings
    across every model type here). Standard Cox/Breslow estimator:
    H0(t) = sum over observed event times t_i<=t of
    d_i / sum_{j in risk set at t_i} r_j, where r_j is subject j's
    (shifted-positive) risk score and the risk set at t_i is everyone with
    duration>=t_i.

    Replaces the old fixed `exp(-risk_norm*t/365)` transform (still used by
    pkgs/experiments/utils.py's compute_brier_score_from_risk_scores for the
    older scenarios' own training/eval scripts -- deliberately NOT touched
    here, out of scope), which assumed every model's risk score, once
    shifted to start at 0.01, IS directly an annual hazard rate. That
    assumption had no grounding: different architectures' raw risk scores
    span wildly different magnitudes (a Cox partial hazard vs. an SVM
    decision score vs. a GBSA score), so the fixed formula saturated to ~0
    survival (~1.0 event probability) for nearly the whole cohort whenever a
    model's shifted score happened to be more than a few units -- see Stage
    2.2 Finding #2, generated_data/rep99/stage2_2_debug_report.txt, for the
    repeated "predicted risk is the same for every patient" symptom this
    caused across 6 of 11 models, every scenario. Fitting H0(t) empirically
    from each model's own training-set risk-score distribution absorbs
    whatever arbitrary scale/units that model's score is in -- no hardcoded
    divisor, no assumption about what "1 unit of risk_score" means in
    calendar time.

    Efficient O(n log n) implementation: sort ascending by duration: the
    risk set at any time t is exactly the suffix of subjects whose duration
    >= t in this sorted order, so a single reverse cumulative sum gives
    every subject's risk-set denominator in one pass, instead of
    recomputing a boolean-mask sum per event time (O(n * n_events)).

    Returns (train_shift, event_times, H0): `train_shift` is the constant
    (train_risk_scores.min()) that must be subtracted from ANY risk score --
    train or test, from this same model -- before combining it with H0;
    `event_times`/`H0` describe a right-continuous step function (flat
    between events, H0[i] is the cumulative hazard at event_times[i])."""
    r = np.asarray(train_risk_scores, dtype=np.float64)
    train_shift = float(r.min())
    r = np.clip(r - train_shift, 1e-6, None)
    d = np.asarray(train_durations, dtype=np.float64)
    e = np.asarray(train_events).astype(bool)

    order = np.argsort(d)
    d_sorted, e_sorted, r_sorted = d[order], e[order], r[order]
    # risk_set_sum[i] = sum(r_sorted[i:]) = total risk score of everyone with
    # duration >= d_sorted[i] (a suffix sum since d_sorted is ascending).
    risk_set_sum = np.cumsum(r_sorted[::-1])[::-1]

    event_times = np.unique(d_sorted[e_sorted])
    H0 = np.empty(len(event_times), dtype=np.float64)
    cumulative = 0.0
    for i, t in enumerate(event_times):
        first_idx = int(np.searchsorted(d_sorted, t, side='left'))
        events_at_t = int(np.sum((d_sorted == t) & e_sorted))
        denom = risk_set_sum[first_idx]
        cumulative += (events_at_t / denom) if denom > 0 else 0.0
        H0[i] = cumulative
    return train_shift, event_times, H0


def _baseline_cumulative_hazard_at(event_times, H0, query_times):
    """Right-continuous step-function lookup: H0 at each query time (0 before
    the first observed event time)."""
    query_times = np.asarray(query_times, dtype=np.float64)
    if len(event_times) == 0:
        return np.zeros_like(query_times)
    idx = np.searchsorted(event_times, query_times, side='right') - 1
    return np.where(idx >= 0, H0[np.clip(idx, 0, len(H0) - 1)], 0.0)


def calibrated_survival_probs(risk_scores, times, baseline):
    """S(t) = exp(-r_shifted * H0(t)), using a per-model Breslow baseline
    hazard (train_shift, event_times, H0) already fit from this SAME model's
    own training-set risk scores via fit_breslow_baseline_hazard(). Test
    risk scores below train_shift (the training set's own minimum) are
    clipped to a small positive floor rather than going negative -- the
    baseline was only ever estimated for non-negative relative risk."""
    train_shift, event_times, H0 = baseline
    risk_scores = np.asarray(risk_scores, dtype=np.float64)
    r_shifted = np.clip(risk_scores - train_shift, 1e-6, None)
    times = np.asarray(times, dtype=np.float64)
    H0_t = _baseline_cumulative_hazard_at(event_times, H0, times)
    return np.exp(-np.outer(r_shifted, H0_t))


def predicted_event_prob_at(risk_scores, horizon_days, baseline):
    """1 - S(horizon) per row, from the per-model calibrated baseline-hazard transform."""
    surv = calibrated_survival_probs(risk_scores, [horizon_days], baseline)[:, 0]
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


def calibration_table(predicted, durations, events, horizon_days, n_bins=10):
    """Per-decile predicted-risk-vs-KM-observed-risk table at one horizon.
    `predicted` is already a per-row/subject predicted probability at
    `horizon_days` — see resolve_predicted_prob() for how callers get it
    (native model output where available, generic transform otherwise)."""
    df = pd.DataFrame({'predicted': predicted, 'duration': durations, 'event': events})

    needs_fallback = False
    try:
        bins = pd.qcut(df['predicted'], q=n_bins, duplicates='drop')
        # A fully degenerate `predicted` (e.g. every row gets exactly 1.0 —
        # observed from an undertrained rep99 ddh model) doesn't raise here;
        # pandas silently returns an all-NaN bin column instead, which then
        # groups into zero groups below with no error at all. Must check for
        # this explicitly rather than relying on the except clause.
        needs_fallback = bins.isna().all()
        df['bin'] = bins
    except ValueError:
        needs_fallback = True

    if needs_fallback:
        # Too few distinct predicted values for n_bins quantile cuts (small
        # rep99 samples, or a model whose risk scores collapsed to near-ties/
        # a single constant value) — rank-based cuts are always distinct, so
        # this always produces n_bins groups (or len(df) if smaller), even
        # when `predicted` itself carries no information at all. In that
        # constant-`predicted` case the resulting decile table is a
        # legitimate finding — "this model draws no distinction between
        # patients at this horizon" — not a plotting artifact to hide.
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


def model_net_benefit_curve(predicted, durations, events, horizon_days, thresholds=DCA_THRESHOLDS):
    """Net benefit at each threshold, per Vickers & Elkin's decision-curve-
    analysis formula, with P(event by horizon | risk>=pt) estimated via
    Kaplan-Meier so censoring before `horizon_days` doesn't bias the count (a
    plain proportion would). `predicted` is already a per-row/subject
    predicted probability at `horizon_days` — see resolve_predicted_prob().
    Rows with a NaN prediction are dropped (a NaN never satisfies `>= pt`, so
    without this they'd silently count as "definitely low-risk" in the
    denominator at every threshold rather than being excluded, biasing net
    benefit down instead of raising or being visibly absent)."""
    predicted = np.asarray(predicted, dtype=np.float64)
    durations = np.asarray(durations)
    events = np.asarray(events)
    valid = ~np.isnan(predicted)
    predicted, durations, events = predicted[valid], durations[valid], events[valid]
    n = len(predicted)

    model_nb = {}
    if n == 0:
        return {pt: None for pt in thresholds}
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


def brier_score_up_to(df_train, durations, events, risk_scores, horizon_days, baseline):
    """Integrated Brier score, 0 to horizon_days. Takes `durations`/`events`
    directly (this model's own prediction-row labels) rather than re-deriving
    them from df_test, as an earlier version did via
    compute_brier_score_from_survival_probs(df_train, df_test, ...) — that
    silently broke once dynamic_deephit/hazard_transformer's predictions were
    fixed to be one-per-SUBJECT (see their *_predictions() docstrings above),
    since df_test is one row per lab EVENT: `Surv.from_dataframe(...,
    data=df_test)` built a y_test of len(df_test) while survival_probs had
    len(unique subjects) rows, so sksurv's shape check failed and this always
    returned None for those two models (caught, not a crash, but silently
    wrong) — always build y_test from the same durations/events actually
    paired with risk_scores instead.

    Also administratively censors any test subject followed longer than
    df_train's longest observed duration, at that train-set max. sksurv's
    IPCW machinery estimates the censoring distribution G(t) from y_train
    alone, so it has no information past y_train's max follow-up; a test
    subject who WAS followed past it (observed: four_features rep1, train
    max 4216d vs test max 4333d) makes sksurv raise "time must be smaller
    than largest observed time point" -- for EVERY horizon_days, not just
    the one requested (confirmed by testing horizons from 1yr to 12yr, all
    identically broken) -- since the break is a train/test max-duration
    mismatch, not a horizon choice. There's no true information available
    about what happens to that subject past the training set's covered
    range, so the standard treatment is to cut their follow-up there and
    mark them censored at that point, same as any other right-censoring."""
    times = np.linspace(1, horizon_days, 50)
    survival_probs = calibrated_survival_probs(risk_scores, times, baseline)
    y_train = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df_train)

    durations = np.asarray(durations, dtype=np.float64).copy()
    events = np.asarray(events).astype(bool).copy()
    train_max = float(df_train['duration_in_days'].max())
    beyond_train_range = durations > train_max
    if beyond_train_range.any():
        durations[beyond_train_range] = train_max
        events[beyond_train_range] = False

    y_test = Surv.from_arrays(event=events, time=durations)
    try:
        return round(float(integrated_brier_score(y_train, y_test, survival_probs, times)), 5)
    except Exception as e:
        print(f"Warning: Could not compute Brier Score: {e}")
        return None


def resolve_predicted_prob(risk_scores, native_prob_fn, horizon_days, baseline):
    """Predicted probability of event by horizon_days, preferring a model's own
    native time-indexed prediction (exact, no extrapolation) over the
    per-model calibrated baseline-hazard transform (approximate, used as a
    fallback). Returns (predicted, source) where source is 'native' or
    'approximate' — logged by the caller so the report says which one
    applies to each number."""
    if native_prob_fn is not None:
        native = native_prob_fn(horizon_days)
        if native is not None:
            return native, 'native'
    return predicted_event_prob_at(risk_scores, horizon_days, baseline), 'approximate'


def discrimination_metrics(df_train, risk_scores, durations, events, baseline, auc_horizon_days=730):
    """C-index, integrated Brier score (0 to auc_horizon_days), and mean
    time-dependent AUC (0 to auc_horizon_days) — one summary triple per model,
    for the cross-model comparison charts. Sign convention: risk_scores is
    "higher = riskier" for every model here (see each *_predictions() function
    above), so concordance_index needs it negated — lifelines expects a score
    that's higher for LONGER survival (confirmed against this codebase's own
    cox.py, which negates its partial-hazard risk score the same way before
    calling concordance_index). `baseline` is this same model's own
    fit_breslow_baseline_hazard() result (None if fitting it failed — brier
    then fails too, caught below and recorded in errors like any other)."""
    result = {'c_index': None, 'brier': None, 'auc': None, 'errors': {}}
    try:
        result['c_index'] = round(float(concordance_index(durations, -np.asarray(risk_scores), events)), 4)
    except Exception as e:
        result['errors']['c_index'] = str(e)

    try:
        result['brier'] = brier_score_up_to(df_train, durations, events, risk_scores, auc_horizon_days, baseline)
    except Exception as e:
        result['errors']['brier'] = str(e)

    try:
        y_train = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df_train)
        max_time = min(float(np.max(durations)), auc_horizon_days - 1)
        if max_time > 1:
            y_test = Surv.from_arrays(event=np.asarray(events).astype(bool), time=np.asarray(durations))
            times = np.arange(1, max(max_time, 2), 1)
            _, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)
            result['auc'] = round(float(mean_auc), 4)
    except Exception as e:
        # sksurv raises "censoring survival function is zero at one or more
        # time points" whenever the test set has no one censored beyond some
        # point in `times` (IPCW weight denominator hits zero) — a known,
        # already-documented edge case elsewhere in this repo (see
        # EXPERIMENT_STATUS.md's Stage 3 notes: "hit the known (unrelated)
        # censoring-edge-case on AUC"), not a bug here. Recorded rather than
        # silently swallowed so a bare `auc=None` in the report is traceable.
        result['errors']['auc'] = str(e)

    return result


# Every model's own "how to turn raw output into (risk_scores, durations,
# events, native_prob_fn)" logic -- including fetching its own train/test
# data (get_train_test_data()/get_last_observation_data(), whichever shape
# it needs) and building its own Dataset/DataLoader/tensor -- is now a
# predictions(scenario, split='test') method on that model's own class in
# pkgs/models/ (deepsurv/dynamicdeephit/hazard_transformer/rnnsurv each
# already had an architecture class there -- predictions() was added
# directly to each, including moving their Dataset classes in from
# pkgs/experiments/; cox/kfre/logistic_hazard/gbsa/srf/survival_svm/weibul
# had no architecture class of their own, since they're direct calls into
# lifelines/pycox/sksurv, so a thin wrapper class was added holding the
# fitted estimator, same predictions()-method shape). See
# pkgs/models/cox.py's module docstring for the history of this split
# (Stage 2.2's refactor) and ClinicalValidityAnalyzer._get_predictions
# below for the (now purely dispatch, no domain logic) call sites.
#
# Each predictions() returns (risk_scores, durations, events, native_prob_fn):
#   - risk_scores: a scalar per row/subject, higher = riskier. Used for ranking
#     metrics (C-index, AUC) and, via the shared calibrated-baseline-hazard
#     transform, as the FALLBACK way to get a predicted probability at an
#     arbitrary horizon when no better option exists.
#   - native_prob_fn(horizon_days) -> array or None: when the model has a real,
#     time-indexed prediction that can be read off AT the requested horizon
#     (not approximated from a generic risk score), this returns it; None means
#     "no native prediction available at this horizon, use the fallback."
#     cox has no such thing, so it always returns None.


class ClinicalValidityAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.current_scenario = None
        self.scenario_report_lines = {}
        self.models = ['cox', 'ddh', 'hazard_transformer', 'logistic_hazard', 'rnn_surv', 'kfre',
                       'deepsurv', 'gbsa', 'srf', 'survival_svm', 'weibul']
        self.model_pretty_names = {
            'cox': 'Cox', 'ddh': 'Dynamic DeepHit', 'hazard_transformer': 'Hazard Transformer',
            'logistic_hazard': 'Logistic Hazard', 'rnn_surv': 'RNN-Surv', 'kfre': 'KFRE',
            'deepsurv': 'DeepSurv', 'gbsa': 'GBSA', 'srf': 'Survival RF',
            'survival_svm': 'Survival SVM', 'weibul': 'Weibull AFT',
        }
        # scenario_name -> horizon_days -> {'treat_all':.., 'egfr_nb':.., 'models': {model_name: {'calibration':.., 'model_nb':..}}}
        # populated by analyze_scenario, read back by create_calibration_plot/create_decision_curve_plot.
        self.all_results = {}

    def log(self, message):
        print(message)
        if self.current_scenario is not None:
            self.scenario_report_lines.setdefault(self.current_scenario, []).append(message)

    def _model_paths(self, scenario_name):
        paths = {
            'cox': generate_data_path_latest_rep + f'/{scenario_name}_cox_model.dill',
            'ddh': generate_data_path_latest_rep + f'/{scenario_name}_ddh_model.pt',
            'hazard_transformer': generate_data_path_latest_rep + f'/{scenario_name}_hazard_transformer_model.pt',
            'logistic_hazard': generate_data_path_latest_rep + f'/{scenario_name}_logistic_hazard_model.pt',
            'rnn_surv': generate_data_path_latest_rep + f'/{scenario_name}_rnn_surv_model.pt',
            'deepsurv': generate_data_path_latest_rep + f'/{scenario_name}_deepsurv_model.pt',
            'gbsa': generate_data_path_latest_rep + f'/{scenario_name}_gbsa_model.dill',
            'srf': generate_data_path_latest_rep + f'/{scenario_name}_srf_model.dill',
            'survival_svm': generate_data_path_latest_rep + f'/{scenario_name}_survival_svm_model.dill',
            'weibul': generate_data_path_latest_rep + f'/{scenario_name}_weibul_model.dill',
        }
        # KFRE has no published equation for twenty_features_heterogeneous (only
        # 4-/8-variable, per kfre.py) — omit the key entirely rather than
        # pointing at a path that can never exist, so the existing "model file
        # not found, skip" path above handles it the same way as any other
        # not-yet-trained model, no special-casing needed.
        if scenario_name in ('four_features', 'eight_features'):
            paths['kfre'] = get_kfre_risk_scores_path(
                ExperimentScenario.FOUR_FEATURES if scenario_name == 'four_features'
                else ExperimentScenario.EIGHT_FEATURES, years=2)
        return paths

    # model_name -> the pkgs/models/ wrapper class to construct around a
    # loaded lifelines/sksurv estimator (deepsurv/dynamicdeephit/
    # hazard_transformer/rnn_surv need no wrapper: torch.load() already
    # returns an instance of their own class, predictions() is a bound
    # method on it directly).
    _SKLEARN_STYLE_MODEL_CLASSES = {
        'cox': CoxModel, 'gbsa': GBSAModel, 'srf': SRFModel,
        'survival_svm': SurvivalSVMModel, 'weibul': WeibulModel,
    }

    def _get_predictions(self, model_name, model_path, scenario, split='test'):
        """Loads the model file, wraps it in its own class from pkgs/models/
        where one is needed, and calls that model's own
        predictions(scenario, split=split). Every model now fetches its own
        data and does its own forward-pass/predict() interpretation (per
        Stage 2.2's model-layer refactor -- see pkgs/models/cox.py's module
        docstring) -- this method is pure dispatch, no domain logic."""
        if model_name in self._SKLEARN_STYLE_MODEL_CLASSES:
            model = load_pkl_and_dill_model(model_path)
            if model is None:
                return None
            model_cls = self._SKLEARN_STYLE_MODEL_CLASSES[model_name]
            return model_cls(model).predictions(scenario, split=split)

        if model_name == 'kfre':
            return KFREModel(scenario).predictions(split=split)

        if model_name == 'logistic_hazard':
            net = torch.load(model_path, map_location='cpu', weights_only=False)
            return LogisticHazardModel(net).predictions(scenario, split=split)

        if model_name in ('hazard_transformer', 'ddh', 'rnn_surv', 'deepsurv'):
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            return model.predictions(scenario, split=split)

        raise ValueError(f"Unknown model_name {model_name}")

    def _get_train_risk_scores(self, model_name, model_path, scenario):
        """Same dispatch as _get_predictions, but scores the TRAINING set
        instead of the test set (split='train'). Used only to fit this
        model's own Breslow baseline hazard (fit_breslow_baseline_hazard)
        -- never reported as a prediction itself. Reloads the model file
        rather than threading an already-loaded object through from
        _get_predictions: these are all small rep99/rep1 checkpoints, so
        the extra I/O is cheap, and it keeps baseline-fitting fully
        independent of the reported test predictions (a bug here can't
        silently corrupt those, or vice versa)."""
        result = self._get_predictions(model_name, model_path, scenario, split='train')
        if result is None:
            raise ValueError(f"No usable model at {model_path}")
        risk_scores, durations, events, _ = result
        return risk_scores, durations, events

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

        # eGFR-threshold referral rule rows: for scenarios where each row is one
        # lab EVENT (e.g. twenty_features_heterogeneous — see
        # time_series_store.py's heterogeneous branch), only the row's own drawn
        # lab has a real value; every other lab column (including egfr) is a
        # placeholder 0 with a companion `<lab>_missing=1` flag. Feeding those
        # placeholder zeros into "egfr < cutoff" misclassifies them as severely
        # low eGFR, inflating n_high toward n and blowing up the net-benefit
        # formula's implied_pt/(1-implied_pt) term (see Stage 2.2 debug report,
        # generated_data/rep99/stage2_2_debug_report.txt, Finding #1 — confirmed
        # 8,613/9,115 rep99 twenty_features_heterogeneous test rows had
        # egfr_missing=1/egfr=0). Restrict to rows with a genuine eGFR
        # measurement when that flag column exists; four_features/eight_features
        # have no egfr_missing column (egfr is always real there, anchored on a
        # creatinine draw) so this is a no-op for them.
        egfr_referral_df = df_test
        if has_egfr and 'egfr_missing' in df_test.columns:
            egfr_referral_df = df_test[df_test['egfr_missing'] == 0]
            self.log(f"eGFR-threshold referral rule: restricting to "
                      f"{len(egfr_referral_df)}/{len(df_test)} rows with a real eGFR "
                      f"measurement (egfr_missing == 0); other rows are placeholder "
                      f"egfr=0 from this scenario's per-lab-event row format.")
            if len(egfr_referral_df) == 0:
                has_egfr = False

        # Get every model's predictions once up front — risk_scores/durations/
        # events don't depend on horizon, only the model does. Also fit each
        # model's own Breslow baseline hazard here (from that SAME model's
        # risk scores on df_train) — used by every "approximate" (no native
        # per-horizon output) probability/Brier calculation below. Kept in a
        # separate try/except from the test-side prediction so a
        # baseline-fitting failure doesn't discard predictions already
        # obtained; downstream code treats a missing baseline the same as
        # any other per-model computation error (caught, logged, that one
        # number reported as unavailable rather than the whole model
        # dropped).
        predictions = {}
        baselines = {}
        for model_name, model_path in self._model_paths(scenario_name).items():
            if not os.path.exists(model_path):
                self.log(f"Model file not found, skipping: {model_path}")
                continue
            try:
                preds = self._get_predictions(model_name, model_path, scenario_enum)
                if preds is None:
                    self.log(f"No usable model at {model_path}, skipping {model_name}.")
                    continue
                predictions[model_name] = preds
            except Exception as e:
                self.log(f"Error getting predictions for {model_name}: {e}")
                continue

            try:
                train_risk_scores, train_durations, train_events = self._get_train_risk_scores(
                    model_name, model_path, scenario_enum)
                baselines[model_name] = fit_breslow_baseline_hazard(
                    train_risk_scores, train_durations, train_events)
            except Exception as e:
                self.log(f"Error fitting baseline hazard for {model_name}: {e}")
                baselines[model_name] = None

        # Discrimination metrics (C-index / Brier / AUC) — one per model, not
        # per horizon (fixed at the 2yr convention used throughout pkgs/experiments/*.py).
        # Used for the cross-scenario comparison charts built after all scenarios run.
        metrics = {}
        self.log("\nDiscrimination metrics (C-index / integrated Brier / mean time-dependent AUC, 0-730d):")
        for model_name, (risk_scores, durations, events, _) in predictions.items():
            try:
                m = discrimination_metrics(df_train, risk_scores, durations, events, baselines.get(model_name))
            except Exception as e:
                m = {'c_index': None, 'brier': None, 'auc': None}
                self.log(f"  Error computing discrimination metrics for {model_name}: {e}")
            metrics[model_name] = m
            self.log(f"  {self.model_pretty_names[model_name]}: c_index={m['c_index']} "
                      f"brier={m['brier']} auc={m['auc']}")
            for metric_name, err in m.get('errors', {}).items():
                self.log(f"    ({metric_name} unavailable: {err})")

        scenario_results = {'horizons': {}, 'metrics': metrics}

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
                        egfr_referral_df['egfr'].values, egfr_referral_df['duration_in_days'].values,
                        egfr_referral_df['has_esrd'].values, horizon)
                    self.log(f"eGFR-threshold referral rule net benefit: {egfr_nb}")
                except Exception as e:
                    self.log(f"Error computing eGFR-threshold net benefit: {e}")

            horizon_result = {'treat_all': treat_all, 'egfr_nb': egfr_nb, 'models': {}}

            for model_name, (risk_scores, durations, events, native_prob_fn) in predictions.items():
                self.log(f"\n--- {self.model_pretty_names[model_name]} ---")

                predicted, source = resolve_predicted_prob(
                    risk_scores, native_prob_fn, horizon, baselines.get(model_name))
                self.log(f"Predicted-probability source: {source} "
                         f"({'model output read directly at this horizon' if source == 'native' else 'per-model calibrated baseline-hazard extrapolation — see module docstring'})")

                predicted = np.asarray(predicted, dtype=np.float64)
                n_nan = int(np.isnan(predicted).sum())
                if n_nan == len(predicted):
                    # All-NaN predictions (e.g. a severely undertrained model —
                    # this repo has documented NaN-loss issues before) silently
                    # produce an empty calibration table with no error either
                    # way (pd.qcut and its rank-based fallback both just return
                    # all-NaN bins, no exception) — the same "silently empty
                    # section, looks broken not wrong" failure mode as the
                    # degenerate-constant case above, just via a different
                    # input. Caught explicitly here instead of downstream.
                    self.log(f"  SKIPPED: all {len(predicted)} predicted values are NaN — "
                              "model produced no usable output at this horizon.")
                    horizon_result['models'][model_name] = {
                        'calibration': None, 'model_nb': None, 'predicted_prob_source': source,
                    }
                    continue
                if n_nan > 0:
                    self.log(f"  NOTE: {n_nan}/{len(predicted)} predicted values are NaN "
                              "and excluded from the calibration/DCA calculations below.")

                table = None
                self.log("Calibration (predicted risk decile vs. KM-observed risk):")
                try:
                    table = calibration_table(predicted, durations, events, horizon)
                    if table and len({row['mean_predicted_risk'] for row in table}) == 1:
                        self.log(f"  NOTE: predicted risk is the same ({table[0]['mean_predicted_risk']}) "
                                  "for every patient at this horizon — this model draws no distinction "
                                  "between patients here; the decile split below is an arbitrary rank "
                                  "tie-break, not a real risk gradient.")
                    for row in table:
                        self.log(f"  n={row['n']:>4} events={row['events']:>3} "
                                  f"predicted={row['mean_predicted_risk']:.4f} "
                                  f"observed(KM)={row['km_observed_risk']}")
                except Exception as e:
                    self.log(f"  Error computing calibration table: {e}")

                try:
                    brier = brier_score_up_to(
                        df_train, durations, events, risk_scores, horizon, baselines.get(model_name))
                    self.log(f"Integrated Brier score (0-{horizon:.0f}d): {brier}")
                except Exception as e:
                    self.log(f"  Error computing Brier score: {e}")

                model_nb = None
                self.log("Decision curve analysis (net benefit by risk threshold):")
                try:
                    model_nb = model_net_benefit_curve(predicted, durations, events, horizon)
                    for pt in DCA_THRESHOLDS:
                        self.log(f"  pt={pt:.2f}  model={model_nb[pt]}  "
                                  f"treat_all={treat_all[pt]}  treat_none=0.0")
                except Exception as e:
                    self.log(f"  Error computing decision curve: {e}")

                horizon_result['models'][model_name] = {
                    'calibration': table, 'model_nb': model_nb, 'predicted_prob_source': source,
                }

            scenario_results['horizons'][horizon] = horizon_result

        self.all_results[scenario_name] = scenario_results
        self.create_calibration_plot(scenario_name, horizons)
        self.create_decision_curve_plot(scenario_name, horizons)
        self.save_scenario_report(scenario_name)

    def create_calibration_plot(self, scenario_name, horizons):
        """<scenario>_calibration_plot.png — rows=horizons, cols=models. Each
        panel: mean predicted risk per decile (x) vs. KM-observed risk per
        decile (y), with a dashed y=x reference diagonal."""
        try:
            results = self.all_results[scenario_name]['horizons']
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
            results = self.all_results[scenario_name]['horizons']
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

    def create_metrics_comparison_charts(self):
        """One PNG per metric (C-index, Brier, AUC), comparing all 5 models
        across every scenario analyzed so far in this run — grouped bar chart,
        x=model, one bar group per scenario. Call after all analyze_*()
        methods so self.all_results has every scenario's 'metrics' dict."""
        metric_specs = [
            ('c_index', 'c_index_comparison.png', 'C-index (higher is better)'),
            ('brier', 'brier_comparison.png', 'Integrated Brier score, 0-730d (lower is better)'),
            ('auc', 'auc_comparison.png', 'Mean time-dependent AUC, 0-730d (higher is better)'),
        ]
        scenarios = [s for s in self.all_results if self.all_results[s].get('metrics')]
        if not scenarios:
            print("No metrics available for comparison charts.")
            return

        scenario_colors = plt.cm.Set2(np.linspace(0, 1, len(scenarios)))
        n_models = len(self.models)
        bar_width = 0.8 / max(len(scenarios), 1)

        for metric_key, filename, ylabel in metric_specs:
            fig, ax = plt.subplots(figsize=(2 * n_models + 2, 6))
            x = np.arange(n_models)

            for i, scenario_name in enumerate(scenarios):
                metrics = self.all_results[scenario_name]['metrics']
                values = [metrics.get(m, {}).get(metric_key) for m in self.models]
                offsets = x - 0.4 + bar_width * (i + 0.5)
                # None (metric couldn't be computed for that model) plots as a
                # zero-height bar rather than being silently dropped, so a gap
                # is visible instead of looking like the model was never there.
                plot_values = [v if v is not None else 0.0 for v in values]
                bars = ax.bar(offsets, plot_values, width=bar_width, label=scenario_name,
                              color=scenario_colors[i])
                for bar, v in zip(bars, values):
                    if v is None:
                        ax.text(bar.get_x() + bar.get_width() / 2, 0.01, 'N/A',
                                ha='center', va='bottom', fontsize=6, rotation=90, color='gray')

            ax.set_xticks(x)
            ax.set_xticklabels([self.model_pretty_names[m] for m in self.models], rotation=20, ha='right')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{ylabel.split(",")[0].split("(")[0].strip()} by model and scenario (rep{current_rep})')
            ax.legend(fontsize=8)
            ax.axhline(0.0, color='black', linewidth=0.8)

            plt.tight_layout()
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Metrics comparison chart saved to: {output_path}")

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
