"""
Stage 2.1 additional analyses: calibration and decision-curve analysis (DCA).

How to run
----------
From the repository root, activate the project Python environment and use
the parameterized runner (this module defines the analyzer, not a CLI)::

    # Clinical validity for rep1, all three scenarios and applicable models:
    python -m pkgs.scripts.run_experiments analyze --reps 1 --analyses clinical_validity

    # Production reps 1-5, with separate timestamped logs per repetition:
    python -m pkgs.scripts.run_experiments analyze --reps all --analyses clinical_validity --log-dir pkgs/scripts/logs

    # Select repetitions, scenarios, and models:
    python -m pkgs.scripts.run_experiments analyze --reps 2 3 --analyses clinical_validity --scenarios four_features eight_features --models cox dynamic_deephit

    # Twenty-feature analysis on the mini-experiment repetition:
    python -m pkgs.scripts.run_experiments analyze --reps 99 --analyses clinical_validity --scenarios twenty_features_heterogeneous

    # Run both clinical validity and feature importance:
    python -m pkgs.scripts.run_experiments analyze --reps all

Options and prerequisites:
- --reps accepts positive repetition numbers; "all" means 1-5, excluding 99.
  When omitted, it uses CKD_REP or defaults to 1. Each rep/analysis runs in a
  separate process so imported data paths stay scoped to that repetition.
- --scenarios defaults to four_features, eight_features, and
  twenty_features_heterogeneous. --models defaults to all applicable models;
  KFRE is available only for four_features/eight_features. CLI model names
  include dynamic_deephit and rnnsurv (displayed as ddh and rnn_surv internally).
- Existing <scenario>_train_data.csv / <scenario>_test_data.csv must be under
  the repetition directory. The runner reuses saved models and trains missing
  selected models before analysis; training failures make the worker fail.
- Add --dry-run to preview commands without executing them.

Outputs under generated_data/rep<N>/:
- <scenario>_clinical_validity_report.txt
- <scenario>_calibration_plot.png and <scenario>_decision_curve_plot.png
- c_index_comparison.png, brier_comparison.png, and auc_comparison.png

Reruns overwrite these report/chart names. Comparison charts cover only the
scenarios/models selected in that run; rerun the full selection for a complete
comparison. --log-dir preserves separate timestamped execution logs.

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
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score, brier_score
from sksurv.nonparametric import CensoringDistributionEstimator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pkgs.commons import (
    current_rep, generate_data_path_latest_rep,
    four_features_train_data_path, four_features_test_data_path,
    eight_features_train_data_path, eight_features_test_data_path,
    twenty_features_heterogeneous_train_data_path, twenty_features_heterogeneous_test_data_path,
)
from pkgs.data_analysis.types import ExperimentScenario
from pkgs.data_analysis.patient_outcomes import (
    patient_level_outcomes, verify_patient_outcomes, align_predictions_to_patients,
    patient_level_egfr,
)
from pkgs.data_analysis import bootstrap_ci
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
    """Net benefit of "treat everyone" at each threshold — computed once,
    independent of any model, from the SAME patient-level terminal outcomes and
    the SAME included patients as the model curves
    (PAPER_GAPS_EXPERIMENT_PLAN.md Gap 5c).

    `durations`/`events` must be one terminal outcome per included patient
    (patient_level_outcomes()), not raw `df_test` rows. Taken off `df_test`, as
    this previously was, the comparator counted lab ROWS: a patient with 40 lab
    draws contributed 40 observations to "treat everyone" while contributing one
    prediction to every model curve, and each of their intermediate visits
    entered as a terminal non-event. On rep99 four_features that is 3,667
    observations at a 9.4% event rate standing in for 345 patients at a 72.5%
    event rate — the comparator and the models were not being scored on the same
    thing, so the resulting net-benefit comparison was not interpretable in
    either direction.

    Correcting the unit does not by itself mean any model beats treat-all; it
    means the two curves are now the same kind of quantity."""
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
    Rows with a nonfinite prediction are dropped (a NaN never satisfies `>= pt`, so
    without this they'd silently count as "definitely low-risk" in the
    denominator at every threshold rather than being excluded, biasing net
    benefit down instead of raising or being visibly absent). The analyzer
    calls decision_curve_comparison first to apply the same exclusions to
    every strategy displayed together."""
    predicted = np.asarray(predicted, dtype=np.float64)
    durations = np.asarray(durations)
    events = np.asarray(events)
    valid = np.isfinite(predicted)
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


def egfr_threshold_net_benefit(egfr_values, durations, events, horizon_days,
                               egfr_cutoffs=EGFR_REFERRAL_CUTOFFS, thresholds=DCA_THRESHOLDS):
    """Net benefit of a fixed eGFR<cutoff referral rule (the non-model comparator
    KFRE's own clinical-utility papers use), evaluated ACROSS THE SAME RISK
    THRESHOLDS as every model curve (PAPER_GAPS_EXPERIMENT_PLAN.md Gap 5d:
    "compare strategies at the same decision thresholds").

    `egfr_values`/`durations`/`events` must be ONE REFERRAL DECISION PER PATIENT
    (patient_level_egfr()), carrying that patient's terminal outcome. Run over
    `egfr_referral_df` rows, as this previously was, it evaluated a per-lab-row
    rule: patients with more eGFR draws were re-decided more often and their
    intermediate visits entered as terminal outcomes, so it was never the same
    evaluation unit as the model curves it was plotted against. Restricting the
    twenty-feature scenario to rows with `egfr_missing == 0` removed the
    placeholder zeros but left the repeated-measurement problem untouched
    (rep99: 502 measured-eGFR rows across 20 patients).

    Patients with no finite measured eGFR at or before their landmark have no
    referral decision. The analyzer intersects eGFR coverage with prediction
    coverage and passes the same cohort to every displayed DCA strategy.

    Threshold handling: an eGFR<cutoff rule is a fixed binary test, so its
    true-positive and false-positive rates do not vary with the risk threshold
    pt -- but its NET BENEFIT does, because pt sets the exchange rate between
    them. The standard decision-curve treatment of a binary test is therefore a
    curve over pt from fixed TP/FP rates: NB(pt) = TPrate - FPrate * pt/(1-pt).
    That is what this returns, so the eGFR rule, the model curves and treat-all
    are all read off the same x-axis.

    This replaces an earlier version that reported ONE number per cutoff, using
    the rule's own referral fraction (n_high/n) as a stand-in "implied
    threshold". That number was not a net benefit at any threshold a model was
    being compared at, so plotting it beside the model curves compared two
    different quantities -- and because the implied threshold moved with the
    referral fraction, the eGFR point also shifted whenever cohort composition
    changed, independently of the rule's actual performance.

    Returns {cutoff: {'n_flagged', 'flagged_fraction', 'tp_rate', 'fp_rate',
    'net_benefit': {pt: value}}}."""
    egfr_values = np.asarray(egfr_values, dtype=np.float64)
    n = len(egfr_values)
    results = {}
    for cutoff in egfr_cutoffs:
        high_risk = egfr_values < cutoff
        n_high = int(high_risk.sum())
        entry = {'n_flagged': n_high, 'flagged_fraction': round(n_high / n, 4) if n else None,
                 'tp_rate': None, 'fp_rate': None,
                 'net_benefit': {pt: None for pt in thresholds}}
        if n == 0:
            results[cutoff] = entry
            continue
        if n_high == 0:
            # Refers nobody: identical to treat-none at every threshold.
            entry['tp_rate'], entry['fp_rate'] = 0.0, 0.0
            entry['net_benefit'] = {pt: 0.0 for pt in thresholds}
            results[cutoff] = entry
            continue
        p_event_given_high = _km_event_prob_at(
            np.asarray(durations)[high_risk], np.asarray(events)[high_risk], horizon_days)
        if p_event_given_high is None:
            results[cutoff] = entry
            continue
        tp_rate = p_event_given_high * (n_high / n)
        fp_rate = (1 - p_event_given_high) * (n_high / n)
        entry['tp_rate'] = round(float(tp_rate), 5)
        entry['fp_rate'] = round(float(fp_rate), 5)
        entry['net_benefit'] = {
            pt: round(float(tp_rate - fp_rate * (pt / (1 - pt))), 5) for pt in thresholds}
        results[cutoff] = entry
    return results


def decision_curve_comparison(predicted_by_model, durations, events, horizon_days,
                              egfr_values=None, thresholds=DCA_THRESHOLDS):
    """All displayed strategies evaluated on one common cohort per horizon.

    Intersect finite predictions from every model with any usable output and,
    when available, finite eGFR values aligned to the same patient positions.
    Entirely unavailable models/rules are omitted explicitly rather than
    emptying the cohort. Partial availability DOES restrict every curve,
    including treat-all. An empty intersection yields unavailable curves.
    Return the mask and coverage so reports and plots expose this selection.
    """
    durations, events = np.asarray(durations), np.asarray(events)
    n = len(durations)
    if durations.shape != (n,) or events.shape != (n,):
        raise ValueError('DCA outcomes must be aligned one-dimensional arrays')
    mask = np.ones(n, dtype=bool)
    usable, unavailable = {}, []
    for name, values in predicted_by_model.items():
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (n,):
            raise ValueError(f'{name}: DCA predictions are not patient-aligned')
        finite = np.isfinite(values)
        if not finite.any():
            unavailable.append(name)
            continue
        usable[name] = values
        mask &= finite
    if egfr_values is not None:
        egfr_values = np.asarray(egfr_values, dtype=np.float64)
        if egfr_values.shape != (n,):
            raise ValueError('DCA eGFR values are not patient-aligned')
        if np.isfinite(egfr_values).any():
            mask &= np.isfinite(egfr_values)
        else:
            egfr_values = None
    d, e = durations[mask], events[mask]
    return {
        'mask': mask, 'n_total': n, 'n_included': int(mask.sum()),
        'unavailable_models': unavailable,
        'treat_all': treat_all_net_benefit_curve(d, e, horizon_days, thresholds),
        'models': {name: model_net_benefit_curve(values[mask], d, e, horizon_days, thresholds)
                   for name, values in usable.items()},
        'egfr_nb': (egfr_threshold_net_benefit(
            egfr_values[mask], d, e, horizon_days, thresholds=thresholds)
            if egfr_values is not None else None),
    }


def build_censoring_reference(train_terminal):
    """The IPCW censoring reference (`y_train`) both the integrated Brier score
    and the time-dependent AUC estimate G(t) from — built from ONE TERMINAL
    OUTCOME PER TRAINING PATIENT (PAPER_GAPS_EXPERIMENT_PLAN.md Gap 5a).

    This replaces `Surv.from_dataframe(event='has_esrd',
    time='duration_in_days', data=df_train)` on the RAW exported frame. That
    frame is one row per lab EVENT, so every intermediate visit of a patient
    still under follow-up entered the reference as a separate TERMINAL censored
    observation at that visit's own duration — inventing censoring that never
    happened and collapsing the reference event rate (rep99 four_features: 5.75%
    across 5,806 training rows vs. 50.0% across the 500 training patients).
    Only the Brier score and the AUC consume this reference; Harrell's C-index
    does not, which is why the C-index moves far less than the other two when
    this is corrected.

    `train_terminal` comes from patient_level_outcomes() and is verified by
    verify_patient_outcomes() before it reaches here."""
    return Surv.from_arrays(
        event=train_terminal['has_esrd'].values.astype(bool),
        time=train_terminal['duration_in_days'].values.astype(float))


def evaluation_time_cap(train_terminal):
    """The latest time the IPCW-weighted metrics can be evaluated at: the
    training maximum, or the floating-point instant just before G(t) reaches
    zero. Use the SAME reverse-KM estimator as sksurv's IPCW metrics: ordinary
    KM on inverted event flags treats event/censoring ties differently.

    sksurv estimates G(t) from y_train and divides by it, so both the
    integrated Brier score and the time-dependent AUC are undefined once G(t)
    hits 0 — sksurv reports this as "censoring survival function is zero at one
    or more time points" and refuses the whole call, for every horizon rather
    than just the one requested, because the break is a train/test follow-up
    mismatch and not a horizon choice. Test patients followed past this cap are
    administratively censored at it (see _administratively_censor).

    The uncapped AUC failed for all models in the September 14 four-feature
    rep99 run; the eight- and twenty-feature runs already returned AUCs.
    G(t) is a right-continuous step function, so follow-up between its last
    positive training knot and its first zero is supported too."""
    durations = train_terminal['duration_in_days'].values.astype(float)
    train_max = float(durations.max())
    censoring = CensoringDistributionEstimator().fit(build_censoring_reference(train_terminal))
    zeros = censoring.unique_time_[censoring.prob_ <= 0]
    cap = min(train_max, float(np.nextafter(zeros[0], -np.inf))) if len(zeros) else train_max
    if not np.isfinite(cap) or cap < 0:
        raise ValueError('Training censoring distribution has no supported nonnegative time')
    return cap, train_max


def survival_prob_matrix(risk_scores, times, baseline, native_prob_fn):
    """(n_patients, len(times)) matrix of predicted SURVIVAL probabilities, plus
    a label saying where it came from (PAPER_GAPS_EXPERIMENT_PLAN.md Gap 5b).

    Preference order:
    1. 'native' — the model's own time-indexed survival output, read at each
       grid time via native_prob_fn (which returns P(event by t); survival is
       1 - that). Used only when the model covers EVERY grid time; a partially
       covered grid would silently mix two different probability definitions
       inside one integral.
    2. 'fitted-conversion' — calibrated_survival_probs(), i.e. this model's
       risk scores pushed through a Breslow baseline fitted from its own
       TRAINING-set scores. This is a real, documented estimator, but it is an
       estimator of the model PLUS that conversion, not of the model's own
       survival probabilities, and every number derived from it is labelled as
       such in the report.

    Returns (matrix, source). matrix is None when neither route is available."""
    times = np.asarray(times, dtype=np.float64)
    if native_prob_fn is not None:
        columns = []
        for t in times:
            native = native_prob_fn(float(t))
            if native is None:
                columns = None
                break
            columns.append(1.0 - np.asarray(native, dtype=np.float64))
        if columns is not None and len(columns) == len(times):
            return np.column_stack(columns), 'native'
    if baseline is None:
        return None, 'unavailable'
    return calibrated_survival_probs(risk_scores, times, baseline), 'fitted-conversion'


def _administratively_censor(durations, events, train_max):
    """Cut any test patient followed longer than the training set's longest
    observed duration back to that maximum, marked censored there.

    sksurv's IPCW machinery estimates the censoring distribution G(t) from
    y_train alone, so it has no information past y_train's max follow-up; a test
    patient who WAS followed past it (observed: four_features rep1, train max
    4216d vs test max 4333d) makes sksurv raise "time must be smaller than
    largest observed time point" for EVERY horizon, not just the one requested
    (confirmed by testing horizons from 1yr to 12yr, all identically broken) —
    the break is a train/test max-duration mismatch, not a horizon choice.
    There is no information available about what happens to that patient past
    the training set's covered range, so the standard treatment is to cut their
    follow-up there and mark them censored, as with any other right-censoring."""
    durations = np.asarray(durations, dtype=np.float64).copy()
    events = np.asarray(events).astype(bool).copy()
    beyond = durations > train_max
    if beyond.any():
        durations[beyond] = train_max
        events[beyond] = False
    return durations, events, int(beyond.sum())


def integrated_brier_up_to(y_train, train_max, durations, events, risk_scores,
                           horizon_days, baseline, native_prob_fn, n_grid=50):
    """Integrated Brier score over 0..horizon_days, preferring the model's own
    survival curve over the fitted risk-score conversion (Gap 5b) and weighted
    by a patient-level censoring reference (Gap 5a).

    Returns {'value', 'source', 'n_censored_at_train_max'}; 'source' is
    'native' or 'fitted-conversion' and is reported next to every value so a
    native IBS is never silently compared against a converted one."""
    times = np.linspace(1, horizon_days, n_grid)
    survival_probs, source = survival_prob_matrix(risk_scores, times, baseline, native_prob_fn)
    if survival_probs is None:
        return {'value': None, 'source': source, 'n_censored_at_train_max': 0}

    durations, events, n_cut = _administratively_censor(durations, events, train_max)
    y_test = Surv.from_arrays(event=events, time=durations)
    try:
        value = round(float(integrated_brier_score(y_train, y_test, survival_probs, times)), 5)
    except Exception as e:
        print(f"Warning: Could not compute Brier Score: {e}")
        value = None
    return {'value': value, 'source': source, 'n_censored_at_train_max': n_cut}


def point_brier_scores(y_train, train_max, durations, events, native_prob_fn, horizons):
    """Brier score AT each supported horizon, rather than integrated over a
    window (PAPER_GAPS_EXPERIMENT_PLAN.md Gap 5b, KFRE row).

    KFRE does not produce a survival CURVE at all: the published equation
    defines S0 only at 2 and 5 years, so those two horizons are the entire
    native output. Integrating over a 0-730d grid would require inventing the
    curve between them, which is exactly the "added survival-curve conversion"
    the gap says must not be passed off as native. A point Brier score at 730d
    and 1825d is the honest version of the same measurement, and it is computed
    here for ANY model with native output at those horizons so KFRE's numbers
    have something comparable to sit beside.

    Returns {horizon: value_or_None}; horizons the model does not natively
    cover are omitted."""
    if native_prob_fn is None:
        return {}
    durations, events, _ = _administratively_censor(durations, events, train_max)
    y_test = Surv.from_arrays(event=events, time=durations)
    results = {}
    for horizon in horizons:
        native = native_prob_fn(float(horizon))
        if native is None:
            continue
        if horizon >= train_max or horizon >= float(np.max(durations)):
            continue
        surv = (1.0 - np.asarray(native, dtype=np.float64)).reshape(-1, 1)
        try:
            _, values = brier_score(y_train, y_test, surv, [float(horizon)])
            results[horizon] = round(float(values[0]), 5)
        except Exception as e:
            print(f"Warning: Could not compute point Brier score at {horizon}d: {e}")
            results[horizon] = None
    return results


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


def discrimination_metrics(y_train, train_max, risk_scores, durations, events, baseline,
                           native_prob_fn=None, auc_horizon_days=730):
    """C-index, integrated Brier score (0 to auc_horizon_days), and mean
    time-dependent AUC (0 to auc_horizon_days) — one summary triple per model,
    for the cross-model comparison charts.

    Sign convention: risk_scores is "higher = riskier" for every model here (see
    each predictions() method in pkgs/models/), so concordance_index needs it
    negated — lifelines expects a score that's higher for LONGER survival
    (confirmed against this codebase's own cox.py, which negates its
    partial-hazard risk score the same way before calling concordance_index).

    `y_train` is the PATIENT-LEVEL censoring reference from
    build_censoring_reference() (Gap 5a); it used to be built from the raw
    row-per-lab-event training frame. `native_prob_fn` lets the Brier score use
    the model's own survival curve where one exists (Gap 5b); `brier_source`
    records which route produced the number. `baseline` is this same model's own
    fit_breslow_baseline_hazard() result, used only when there is no native
    curve (None if fitting it failed — the converted Brier then fails too,
    caught below and recorded in errors like any other)."""
    result = {'c_index': None, 'brier': None, 'brier_source': None,
              'brier_converted': None, 'auc': None, 'point_brier': {}, 'errors': {}}
    ranking_issue = bootstrap_ci.discrimination_unavailable_reason(risk_scores)
    try:
        if ranking_issue:
            raise ValueError(ranking_issue)
        result['c_index'] = round(float(concordance_index(durations, -np.asarray(risk_scores), events)), 4)
    except Exception as e:
        result['errors']['c_index'] = str(e)

    try:
        brier = integrated_brier_up_to(y_train, train_max, durations, events, risk_scores,
                                       auc_horizon_days, baseline, native_prob_fn)
        result['brier'] = brier['value']
        result['brier_source'] = brier['source']
    except Exception as e:
        result['errors']['brier'] = str(e)

    # Always compute the fitted-conversion IBS alongside a native one, so the
    # report can show what the conversion was contributing rather than just
    # swapping one number for another (Gap 5b: "distinguish native-probability
    # metrics from metrics using the fitted conversion in the reports").
    if result['brier_source'] == 'native':
        try:
            converted = integrated_brier_up_to(y_train, train_max, durations, events, risk_scores,
                                               auc_horizon_days, baseline, None)
            result['brier_converted'] = converted['value']
        except Exception as e:
            result['errors']['brier_converted'] = str(e)

    try:
        result['point_brier'] = point_brier_scores(
            y_train, train_max, durations, events, native_prob_fn, DEFAULT_HORIZONS_DAYS)
    except Exception as e:
        result['errors']['point_brier'] = str(e)

    try:
        if ranking_issue:
            raise ValueError(ranking_issue)
        # Same administrative censoring the Brier path applies. Without it the
        # AUC call sees test follow-up past the training censoring
        # distribution's support and sksurv refuses the whole call — which is
        # why all models in the September 14 four-feature rep99 run returned
        # None (the other two scenarios already returned AUCs).
        auc_durations, auc_events, _ = _administratively_censor(durations, events, train_max)
        max_time = min(float(np.max(auc_durations)), auc_horizon_days - 1)
        if max_time > 1:
            y_test = Surv.from_arrays(event=auc_events, time=auc_durations)
            times = np.arange(1, max(max_time, 2), 1)
            _, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)
            result['auc'] = round(float(mean_auc), 4)
    except Exception as e:
        # Preserve the reason for unsupported horizons, undefined rankings,
        # or other metric failures rather than leaving an unexplained None.
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
        # scenario_name -> {'horizons': {horizon_days: {'treat_all': {pt: nb},
        #                                'treat_all_row_level': {pt: nb} (superseded, kept for contrast),
        #                                'egfr_nb': {cutoff: {'n_flagged','flagged_fraction','tp_rate',
        #                                                     'fp_rate','net_benefit': {pt: nb}}},
        #                                'models': {model_name: {'calibration':.., 'model_nb':..,
        #                                                        'predicted_prob_source':..}}}},
        #                  'metrics': {model_name: ..}, 'patient_counts': .., 'egfr_comparator': ..,
        #                  'bootstrap': ..}
        # populated by analyze_scenario, read back by create_calibration_plot/create_decision_curve_plot.
        self.all_results = {}

    def log(self, message):
        message = message.rstrip()
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
        return {model: path for model, path in paths.items() if model in self.models}

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

        # ------------------------------------------------------------------
        # Patient-level evaluation frame (PAPER_GAPS_EXPERIMENT_PLAN.md Gap 5a)
        # ------------------------------------------------------------------
        # Every model produces one prediction per patient. Everything the
        # predictions are scored AGAINST is now built from one terminal outcome
        # per patient too: the IPCW censoring reference, the treat-all curve
        # (Gap 5c) and the eGFR referral comparator (Gap 5d). The raw
        # row-per-lab-event frames are kept only for reporting the contrast.
        train_terminal = patient_level_outcomes(df_train)
        test_terminal = patient_level_outcomes(df_test)
        train_check = verify_patient_outcomes(df_train, train_terminal)
        test_check = verify_patient_outcomes(df_test, test_terminal)

        self.log("Evaluation unit: one terminal outcome per patient (Gap 5a/5c/5d).")
        for split_name, check in (('train', train_check), ('test', test_check)):
            self.log(f"  {split_name}: {check['n_rows']} lab-event rows -> {check['n_patients']} patients; "
                     f"event rate {check['row_event_rate']:.4f} (row-level) vs "
                     f"{check['patient_event_rate']:.4f} (patient-level, {check['n_events']} events)")
            if check['problems']:
                for problem in check['problems']:
                    self.log(f"  WARNING {split_name}: {problem}")
            else:
                self.log(f"  {split_name}: patient uniqueness and terminal-outcome consistency verified.")

        y_train = build_censoring_reference(train_terminal)
        train_max, train_observed_max = evaluation_time_cap(train_terminal)
        test_durations = test_terminal['duration_in_days'].values.astype(float)
        test_events = test_terminal['has_esrd'].values.astype(int)

        max_followup = float(test_terminal['duration_in_days'].max())
        horizons = [h for h in DEFAULT_HORIZONS_DAYS if h < max_followup]
        if not horizons:
            horizons = [max_followup * 0.5]
        self.log(f"Max test follow-up: {max_followup:.1f} days. Horizons used: {horizons}")
        self.log(f"Training-set max follow-up: {train_observed_max:.1f} days; IPCW evaluation capped at "
                 f"{train_max:.1f} days (censoring distribution support). Test patients followed past "
                 "the cap are administratively censored there.")
        self.log("")

        # eGFR-threshold referral rule: one decision per patient at that
        # patient's own prediction landmark, using their most recent genuinely
        # measured eGFR (Gap 5d). In twenty_features_heterogeneous each row
        # carries only the one lab actually drawn, with the rest set to a
        # placeholder 0 plus `<lab>_missing=1`, so most rows carry no eGFR at
        # all and the landmark row itself often does not either.
        egfr_frame, egfr_info = patient_level_egfr(df_test, test_terminal)
        has_egfr = egfr_frame is not None and len(egfr_frame) > 0
        if egfr_frame is None:
            self.log(f"eGFR-threshold referral rule unavailable: {egfr_info.get('reason')}")
        else:
            self.log(f"eGFR-threshold referral rule: one decision per patient from the most recent "
                     f"measured eGFR at or before the landmark — "
                     f"{egfr_info['patients_with_egfr']}/{egfr_info['patients_total']} patients have one "
                     f"({egfr_info['patients_without_egfr']} excluded, no measured eGFR); "
                     f"selected from {egfr_info['rows_with_measured_egfr']}/{egfr_info['rows_considered']} "
                     f"rows with a real eGFR measurement.")

        # Get every model's predictions once up front — risk_scores/durations/
        # events don't depend on horizon, only the model does. Also fit each
        # model's own Breslow baseline hazard here (from that SAME model's risk
        # scores on df_train) — used by every probability/Brier calculation that
        # has no native survival curve to read instead (Gap 5b). Kept in a
        # separate try/except from the test-side prediction so a baseline-fitting
        # failure doesn't discard predictions already obtained.
        predictions = {}
        baselines = {}
        alignment = {}
        for model_name, model_path in self._model_paths(scenario_name).items():
            if not os.path.exists(model_path):
                self.log(f"Model file not found, skipping: {model_path}")
                continue
            try:
                preds = self._get_predictions(model_name, model_path, scenario_enum)
                if preds is None:
                    self.log(f"No usable model at {model_path}, skipping {model_name}.")
                    continue
            except Exception as e:
                self.log(f"Error getting predictions for {model_name}: {e}")
                continue

            # Gap 5a's "verify patient uniqueness and outcome consistency":
            # confirm this model's own durations/events really are the
            # scenario's patient-level terminal outcomes in the canonical
            # subject_id order before pairing its risk scores with anything
            # patient-level. A model that fails this is dropped rather than
            # silently scored against another patient's outcome.
            ok, message = align_predictions_to_patients(
                test_terminal, preds[1], preds[2], self.model_pretty_names[model_name])
            alignment[model_name] = (ok, message)
            if not ok:
                self.log(f"  EXCLUDED — {message}")
                continue
            # Score every model against the SAME canonical patient-level
            # terminal outcomes, keeping only its risk scores and its native
            # survival curve. Models that discretize time (Dynamic-DeepHit and
            # Hazard Transformer floor durations to whole days) or floor a zero
            # duration (Weibull AFT) otherwise carry slightly different
            # outcomes into the comparison than the other models — same
            # patients, but not the same numbers. align_predictions_to_patients
            # has just confirmed the substitution is safe and reported the size
            # of the differences.
            predictions[model_name] = (preds[0], test_durations, test_events, preds[3])

            try:
                train_risk_scores, train_durations, train_events = self._get_train_risk_scores(
                    model_name, model_path, scenario_enum)
                baselines[model_name] = fit_breslow_baseline_hazard(
                    train_risk_scores, train_durations, train_events)
            except Exception as e:
                self.log(f"Error fitting baseline hazard for {model_name}: {e}")
                baselines[model_name] = None

        self.log("\nPatient-level alignment check (Gap 5a):")
        for model_name, (ok, message) in alignment.items():
            self.log(f"  {'OK  ' if ok else 'FAIL'} {message}")

        # Discrimination metrics (C-index / integrated Brier / mean
        # time-dependent AUC) — one per model, at the 2yr convention used
        # throughout pkgs/experiments/*.py. Used for the cross-scenario
        # comparison charts built after all scenarios run.
        metrics = {}
        self.log("\nDiscrimination metrics (C-index / integrated Brier / mean time-dependent AUC, 0-730d):")
        self.log("  Brier source: 'native' = the model's own survival curve; "
                 "'fitted-conversion' = its risk scores pushed through a Breslow baseline fitted "
                 "from its own training scores, i.e. the model PLUS that conversion (Gap 5b).")
        for model_name, (risk_scores, durations, events, native_prob_fn) in predictions.items():
            try:
                m = discrimination_metrics(y_train, train_max, risk_scores, durations, events,
                                           baselines.get(model_name), native_prob_fn)
            except Exception as e:
                m = {'c_index': None, 'brier': None, 'brier_source': None, 'brier_converted': None,
                     'auc': None, 'point_brier': {}, 'errors': {}}
                self.log(f"  Error computing discrimination metrics for {model_name}: {e}")
            metrics[model_name] = m
            self.log(f"  {self.model_pretty_names[model_name]}: c_index={m['c_index']} "
                     f"brier={m['brier']} ({m['brier_source']}) auc={m['auc']}")
            if m.get('brier_converted') is not None:
                self.log(f"    fitted-conversion Brier for the same model/window: {m['brier_converted']} "
                         "(reported for contrast; the native number above is the one to use)")
            if m.get('point_brier'):
                points = ', '.join(f"{int(h)}d={v}" for h, v in sorted(m['point_brier'].items()))
                self.log(f"    native point Brier at published horizons: {points}")
            for metric_name, err in m.get('errors', {}).items():
                self.log(f"    ({metric_name} unavailable: {err})")

        scenario_results = {'horizons': {}, 'metrics': metrics,
                            'patient_counts': {'train': train_check, 'test': test_check},
                            'egfr_comparator': egfr_info}

        # Uncertainty quantification (Gap 3) — patient bootstrap, paired
        # comparisons on the same resamples.
        scenario_results['bootstrap'] = self.run_bootstrap(
            scenario_name, y_train, train_max, test_terminal, predictions, baselines, metrics)

        for horizon in horizons:
            self.log(f"\n=== Horizon: {horizon:.0f} days ===")

            horizon_predictions = {}
            for model_name, (risk_scores, _, _, native_prob_fn) in predictions.items():
                try:
                    predicted, source = resolve_predicted_prob(
                        risk_scores, native_prob_fn, horizon, baselines.get(model_name))
                    predicted = np.asarray(predicted, dtype=np.float64)
                    if predicted.shape != test_durations.shape:
                        raise ValueError('predicted probabilities are not patient-aligned')
                except Exception as exc:
                    self.log(f"  {self.model_pretty_names[model_name]}: horizon prediction unavailable: {exc}")
                    predicted, source = np.full(len(test_terminal), np.nan), 'unavailable'
                horizon_predictions[model_name] = (predicted, source)

            # Reindex by subject ID before taking the common availability mask.
            # The eGFR frame contains only patients with a finite measurement.
            aligned_egfr = (egfr_frame.set_index('subject_id')['egfr'].reindex(
                test_terminal['subject_id']).to_numpy() if has_egfr else None)
            dca = decision_curve_comparison(
                {name: values[0] for name, values in horizon_predictions.items()},
                test_durations, test_events, horizon, aligned_egfr)
            self.log(f"DCA common cohort: {dca['n_included']}/{dca['n_total']} patients "
                     "with finite predictions for every available model"
                     f"{' and measured eGFR' if has_egfr else ''}; "
                     f"{dca['n_total'] - dca['n_included']} excluded from ALL strategies.")
            for name in dca['unavailable_models']:
                self.log(f"  DCA omitted {self.model_pretty_names[name]}: no finite predictions at this horizon.")
            if not dca['n_included']:
                self.log("  DCA unavailable: no patients in the shared cohort.")
            treat_all = dca['treat_all']
            self.log(f"Treat-all net benefit ({dca['n_included']} patients): {treat_all}")
            treat_all_rows = treat_all_net_benefit_curve(
                df_test['duration_in_days'].values, df_test['has_esrd'].values, horizon)
            self.log(f"  (for contrast, the superseded row-level treat-all over "
                     f"{len(df_test)} lab rows: {treat_all_rows})")

            egfr_nb = dca['egfr_nb']
            if egfr_nb is not None:
                for cutoff, entry in egfr_nb.items():
                    self.log(f"eGFR<{cutoff} referral rule "
                             f"({dca['n_included']} patients): flags "
                             f"{entry['n_flagged']} ({entry['flagged_fraction']}), "
                             f"TP rate {entry['tp_rate']}, FP rate {entry['fp_rate']}")
                    self.log(f"  net benefit across the model thresholds: {entry['net_benefit']}")

            horizon_result = {'treat_all': treat_all, 'treat_all_row_level': treat_all_rows,
                              'egfr_nb': egfr_nb, 'models': {},
                              'dca_n_patients': dca['n_included'],
                              'dca_subject_ids': test_terminal.loc[dca['mask'], 'subject_id'].tolist()}

            for model_name, (risk_scores, durations, events, native_prob_fn) in predictions.items():
                self.log(f"\n--- {self.model_pretty_names[model_name]} ---")

                predicted, source = horizon_predictions[model_name]
                self.log(f"Predicted-probability source: {source} "
                         f"({'model output read directly at this horizon' if source == 'native' else 'per-model calibrated baseline-hazard extrapolation — see module docstring'})")

                finite = np.isfinite(predicted)
                n_missing = int((~finite).sum())
                if n_missing == len(predicted):
                    self.log(f"  SKIPPED: all {len(predicted)} predicted values are nonfinite — "
                             "model produced no usable output at this horizon.")
                    horizon_result['models'][model_name] = {
                        'calibration': None, 'model_nb': None, 'predicted_prob_source': source,
                    }
                    continue
                if n_missing > 0:
                    self.log(f"  NOTE: {n_missing}/{len(predicted)} predicted values are nonfinite "
                             "and excluded from calibration. DCA uses the shared cohort above.")

                table = None
                self.log("Calibration (predicted risk decile vs. KM-observed risk):")
                try:
                    table = calibration_table(predicted[finite], durations[finite], events[finite], horizon)
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
                    brier = integrated_brier_up_to(
                        y_train, train_max, durations, events, risk_scores, horizon,
                        baselines.get(model_name), native_prob_fn)
                    self.log(f"Integrated Brier score (0-{horizon:.0f}d): {brier['value']} "
                             f"[{brier['source']}]")
                    if brier['n_censored_at_train_max']:
                        self.log(f"  ({brier['n_censored_at_train_max']} test patients "
                                 f"administratively censored at the training-set max follow-up)")
                except Exception as e:
                    self.log(f"  Error computing Brier score: {e}")

                model_nb = None
                self.log("Decision curve analysis (net benefit by risk threshold):")
                try:
                    model_nb = dca['models'][model_name]
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

    def run_bootstrap(self, scenario_name, y_train, train_max, test_terminal,
                      predictions, baselines, metrics, auc_horizon_days=730):
        """Patient-level bootstrap intervals and paired model comparisons
        (PAPER_GAPS_EXPERIMENT_PLAN.md Gap 3).

        Predictions are computed once by analyze_scenario and only re-indexed
        here, so the whole section costs array indexing plus metric evaluation.
        Every model is scored on the SAME resamples (one shared index matrix),
        which is what makes the paired differences against KFRE and against the
        best model meaningful rather than two independent intervals subtracted.

        Reference models, per Gap 3:
        - KFRE, where the scenario has one (four/eight features only), as the
          published-equation comparator;
        - the best model on the real cohort by C-index, as the internal
          comparator. When KFRE is itself the best model the two coincide and
          only one table is printed."""
        n_patients = len(test_terminal)
        if not predictions or n_patients < 20:
            self.log("\nBootstrap uncertainty: skipped (no usable predictions, or too few patients).")
            return None

        n_bootstrap = int(os.environ.get('CKD_N_BOOTSTRAP', bootstrap_ci.DEFAULT_N_BOOTSTRAP))
        indices = bootstrap_ci.bootstrap_indices(n_patients, n_bootstrap)
        times = np.linspace(1, auc_horizon_days, 50)
        # Bound the AUC grid by the follow-up that actually reaches the metric —
        # i.e. AFTER administrative censoring at the IPCW cap, not the test set's
        # raw maximum, which can exceed it.
        censored_durations, _, _ = _administratively_censor(
            test_terminal['duration_in_days'].values, test_terminal['has_esrd'].values, train_max)
        auc_max = min(float(np.max(censored_durations)), auc_horizon_days - 1)
        auc_times = bootstrap_ci.bootstrap_auc_grid(auc_max)

        self.log(f"\n{'=' * 80}")
        self.log(f"BOOTSTRAP UNCERTAINTY — {n_bootstrap} resamples of the {n_patients} test patients, "
                 "95% percentile intervals (Gap 3)")
        self.log("=" * 80)
        self.log("Only test patients are resampled; the training-set censoring reference is held "
                 "fixed (see pkgs/data_analysis/bootstrap_ci.py for why).")
        self.log("C-index resamples retain full follow-up; IBS/AUC alone use the IPCW-capped outcomes. "
                 "Numerically constant rankings are withheld from points, intervals and paired differences.")
        self.log(f"Bootstrap AUC uses a {len(auc_times)}-point grid over 0-{auc_max:.0f}d rather than "
                 "the per-day grid of the point estimate — see bootstrap_ci.py.")

        replicates = {}
        summaries = {}
        for model_name, (risk_scores, durations, events, native_prob_fn) in predictions.items():
            survival_probs, brier_source = survival_prob_matrix(
                risk_scores, times, baselines.get(model_name), native_prob_fn)
            d, e, _ = _administratively_censor(durations, events, train_max)
            replicates[model_name] = bootstrap_ci.bootstrap_model_metrics(
                y_train, durations, events, risk_scores, survival_probs, times, auc_times, indices,
                ipcw_durations=d, ipcw_events=e)
            point = metrics.get(model_name, {})
            summaries[model_name] = {
                'brier_source': brier_source,
                'c_index': bootstrap_ci.summarize(point.get('c_index'),
                                                  replicates[model_name]['c_index']),
                'brier': bootstrap_ci.summarize(point.get('brier'),
                                                replicates[model_name]['brier']),
                'auc': bootstrap_ci.summarize(point.get('auc'), replicates[model_name]['auc']),
            }

        for metric_key, label in (('c_index', 'C-index'), ('brier', 'Integrated Brier'),
                                  ('auc', 'Mean time-dependent AUC')):
            self.log(f"\n{label} — estimate [95% CI]:")
            for model_name in predictions:
                summary = summaries[model_name][metric_key]
                suffix = (f"  (Brier source: {summaries[model_name]['brier_source']})"
                          if metric_key == 'brier' else "")
                reason = metrics.get(model_name, {}).get('errors', {}).get(metric_key)
                if summary['estimate'] is None and reason:
                    self.log(f"  {self.model_pretty_names[model_name]:<20} n/a [{reason}]{suffix}")
                elif summary['ci_low'] is None:
                    self.log(f"  {self.model_pretty_names[model_name]:<20} {summary['estimate']} "
                             f"[interval unavailable: only {summary['n_bootstrap_usable']} usable "
                             f"resamples]{suffix}")
                else:
                    self.log(f"  {self.model_pretty_names[model_name]:<20} {summary['estimate']} "
                             f"[{summary['ci_low']}, {summary['ci_high']}]"
                             f"  (n={summary['n_bootstrap_usable']}){suffix}")

        # Paired differences on the same resamples.
        reference_models = []
        if 'kfre' in predictions:
            reference_models.append(('kfre', 'KFRE (published equation)'))
        ranked = [(m, metrics.get(m, {}).get('c_index')) for m in predictions]
        ranked = [(m, c) for m, c in ranked if c is not None]
        best_model = max(ranked, key=lambda item: item[1])[0] if ranked else None
        if best_model is not None and best_model not in [m for m, _ in reference_models]:
            reference_models.append((best_model, f'best model by C-index ({self.model_pretty_names[best_model]})'))

        comparisons = {}
        for reference, reference_label in reference_models:
            self.log(f"\nPaired differences vs. {reference_label}, same resamples "
                     "(positive = the listed model is higher):")
            self.log("  C-index and AUC: higher is better. Integrated Brier: LOWER is better, so a "
                     "negative difference favours the listed model.")
            comparisons[reference] = {}
            for model_name in predictions:
                if model_name == reference:
                    continue
                row = {}
                for metric_key in ('c_index', 'brier', 'auc'):
                    row[metric_key] = bootstrap_ci.paired_difference(
                        replicates[model_name][metric_key], replicates[reference][metric_key])
                comparisons[reference][model_name] = row
                parts = []
                for metric_key, label in (('c_index', 'C-index'), ('brier', 'Brier'), ('auc', 'AUC')):
                    diff = row[metric_key]
                    if diff is None:
                        parts.append(f"{label}: n/a")
                    else:
                        marker = '*' if diff['excludes_zero'] else ' '
                        parts.append(f"{label}: {diff['mean_difference']:+.4f} "
                                     f"[{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]{marker}")
                self.log(f"  {self.model_pretty_names[model_name]:<20} " + "   ".join(parts))
            self.log("  * interval excludes 0. These are 95% percentile intervals on paired "
                     "differences, not multiplicity-controlled tests across the model set.")

        return {'n_bootstrap': n_bootstrap, 'n_patients': n_patients,
                'summaries': summaries, 'comparisons': comparisons,
                'reference_models': [m for m, _ in reference_models]}

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
        treat-none and eGFR-cutoff curves, all on the same patient cohort."""
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

                # The eGFR rule is now a CURVE over the same thresholds as the
                # models (Gap 5d), not a single point at an implied threshold of
                # its own, so it is plotted as a line on the same x-axis.
                egfr_nb = horizon_result.get('egfr_nb')
                if egfr_nb:
                    for style, (cutoff, entry) in zip(['-.', ':'], sorted(egfr_nb.items())):
                        curve = entry.get('net_benefit', {}) if isinstance(entry, dict) else {}
                        xy = [(pt, curve[pt]) for pt in DCA_THRESHOLDS if curve.get(pt) is not None]
                        if xy:
                            xs, ys = zip(*xy)
                            ax.plot(xs, ys, style, color='red', linewidth=1.2,
                                    label=f'eGFR<{cutoff} referral')

                ax.set_title(f"{horizon:.0f}-day horizon (n={horizon_result['dca_n_patients']})")
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
