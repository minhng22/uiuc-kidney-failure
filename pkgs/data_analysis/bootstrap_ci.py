"""Patient-level bootstrap confidence intervals and paired model comparisons
for the clinical-validity metrics (PAPER_GAPS_EXPERIMENT_PLAN.md Gap 3).

What is resampled, and what is not
----------------------------------
Only the TEST patients are resampled, with replacement, B times. Each model's
predictions are computed once by the analyzer and then re-indexed per resample
(risk scores, terminal durations/events and a survival-probability matrix on a
fixed time grid are all arrays of length n_patients), so the whole procedure is
array indexing plus metric evaluation — no re-prediction, no retraining.

The IPCW censoring reference (`y_train`) is NOT resampled. It is estimated from
the training cohort, which is fixed across resamples; resampling it as well
would mix training-sample uncertainty into an interval that is meant to express
"how much of this number is the luck of which patients landed in the test
split". The interval reported here is therefore about the test cohort only, and
does not cover uncertainty in the fitted models or in the censoring reference.

Paired comparisons use the SAME resample for both models (Gap 3's "take paired
differences against the Kidney Failure Risk Equation and against the best model
on the same resamples"), so the interval on the difference accounts for the
correlation between two models scored on one cohort. A difference interval that
excludes 0 is evidence the ordering is not resampling noise; it is not a
hypothesis test with a controlled type-I error rate across the 11 models, and
the report says so rather than implying multiplicity was handled.

Cost note: the time-dependent AUC is evaluated on a COARSE grid
(BOOTSTRAP_AUC_GRID_POINTS points spanning the same window) rather than the
one-point-per-day grid the headline number uses. A per-day grid over 730 days
times B resamples times 11 models is the only genuinely expensive part of this
module. Measured on rep99 four_features, the 25-point grid reproduces the
per-day mean AUC to within 0.0003 for Cox (0.59167 vs 0.59196) and Survival RF
(0.58012 vs 0.58019) — two orders of magnitude below the bootstrap interval's
own half-width of about 0.06 — so the grid choice does not move the interval.
The report labels the bootstrap AUC as coarse-grid regardless.
"""
import numpy as np
from lifelines.utils import concordance_index
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score

DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_SEED = 20260918
BOOTSTRAP_AUC_GRID_POINTS = 25


def discrimination_unavailable_reason(risk_scores):
    """Reject rankings that cannot be resolved at the scores' stored precision.

    Check the original dtype BEFORE promoting to float64. A float32 model whose
    entire score range is only a few rounding units can change rankings with
    CPU thread count (observed for rep99's twenty-feature Hazard Transformer).
    We withhold rank metrics when the whole cohort's range is <= 8 * eps *
    max(abs(score)), rather than interpret those differences as discrimination.
    This is a numerical screening convention, not a clinical effect threshold;
    it does not alter the probabilities used for Brier scores or DCA.
    """
    scores = np.asarray(risk_scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        return 'risk scores must be a nonempty, finite one-dimensional array'
    eps = np.finfo(scores.dtype).eps if np.issubdtype(scores.dtype, np.floating) else 0.0
    values = scores.astype(np.float64)
    span = float(np.ptp(values))
    tolerance = 8 * eps * float(np.max(np.abs(values)))
    if span <= tolerance:
        return (f'numerically constant risk scores (range={span:.6g}, '
                f'8*eps*scale={tolerance:.6g}, dtype={scores.dtype}); '
                'ranking metrics withheld; probability metrics remain available')
    return None


def bootstrap_indices(n_patients, n_bootstrap=DEFAULT_N_BOOTSTRAP, seed=DEFAULT_SEED):
    """One (n_bootstrap, n_patients) integer matrix of patient positions drawn
    with replacement. Generated once and shared by every model so that paired
    differences are taken on identical resamples."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_patients, size=(n_bootstrap, n_patients))


def _c_index(durations, events, risk_scores):
    # "higher = riskier" everywhere in this codebase; lifelines wants a score
    # that is higher for LONGER survival, hence the negation (same convention
    # as clinical_validity_analysis.discrimination_metrics).
    return float(concordance_index(durations, -risk_scores, events))


def _ibs(y_train, durations, events, survival_probs, times):
    y_test = Surv.from_arrays(event=events.astype(bool), time=durations)
    return float(integrated_brier_score(y_train, y_test, survival_probs, times))


def _mean_auc(y_train, durations, events, risk_scores, auc_times):
    y_test = Surv.from_arrays(event=events.astype(bool), time=durations)
    _, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, auc_times)
    return float(mean_auc)


def metrics_on_resample(idx, y_train, durations, events, risk_scores,
                        survival_probs, times, auc_times, *,
                        ipcw_durations=None, ipcw_events=None,
                        discrimination_available=True):
    """C-index / integrated Brier / mean time-dependent AUC for one resample of
    patient positions. Each metric is computed inside its own try/except: a
    resample can legitimately break one of them without breaking the others
    (e.g. no cases/controls at an AUC time, or resampled follow-up shorter than
    the fixed integration grid). Record that metric as unavailable on the
    draw, retaining the others. Reports expose the surviving draw count;
    intervals with failed draws are conditional on metric availability."""
    d, e, r = durations[idx], events[idx], risk_scores[idx]
    # Harrell's C uses the original follow-up; only the IPCW metrics use the
    # administratively censored copy. Both copies share the same patient draw.
    wd = d if ipcw_durations is None else ipcw_durations[idx]
    we = e if ipcw_events is None else ipcw_events[idx]
    out = {'c_index': None, 'brier': None, 'auc': None}
    if discrimination_available:
        try:
            out['c_index'] = _c_index(d, e, r)
        except Exception:
            pass
    if survival_probs is not None:
        try:
            out['brier'] = _ibs(y_train, wd, we, survival_probs[idx], times)
        except Exception:
            pass
    if discrimination_available and auc_times is not None and len(auc_times):
        try:
            out['auc'] = _mean_auc(y_train, wd, we, r, auc_times)
        except Exception:
            pass
    return out


def bootstrap_model_metrics(y_train, durations, events, risk_scores, survival_probs,
                            times, auc_times, indices, *,
                            ipcw_durations=None, ipcw_events=None):
    """Per-metric arrays of bootstrap replicates (NaN where that draw failed),
    aligned row-for-row with `indices` so two models' replicate arrays can be
    subtracted directly for a paired comparison. `durations/events` are always
    the original outcomes; optional `ipcw_*` are the capped copies for IBS/AUC.
    Numerically unresolved rankings are withheld for the whole cohort before
    resampling, consistently with the point estimates."""
    discrimination_available = discrimination_unavailable_reason(risk_scores) is None
    durations = np.asarray(durations, dtype=np.float64)
    events = np.asarray(events).astype(bool)
    risk_scores = np.asarray(risk_scores, dtype=np.float64)
    if (ipcw_durations is None) != (ipcw_events is None):
        raise ValueError('IPCW durations and events must be supplied together')
    if ipcw_durations is not None:
        ipcw_durations = np.asarray(ipcw_durations, dtype=np.float64)
        ipcw_events = np.asarray(ipcw_events).astype(bool)
        if ipcw_durations.shape != durations.shape or ipcw_events.shape != events.shape:
            raise ValueError('IPCW outcomes must have the same patient positions as original outcomes')
    replicates = {'c_index': [], 'brier': [], 'auc': []}
    for idx in indices:
        m = metrics_on_resample(idx, y_train, durations, events, risk_scores,
                                survival_probs, times, auc_times,
                                ipcw_durations=ipcw_durations, ipcw_events=ipcw_events,
                                discrimination_available=discrimination_available)
        for key in replicates:
            replicates[key].append(np.nan if m[key] is None else m[key])
    return {key: np.asarray(values, dtype=np.float64) for key, values in replicates.items()}


def percentile_ci(replicates, alpha=0.05):
    """(lower, upper, n_usable) percentile interval over the non-NaN replicates,
    or (None, None, n_usable) when too few draws survived to describe one."""
    values = np.asarray(replicates, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 20:
        return None, None, int(len(values))
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), int(len(values))


def summarize(point_estimate, replicates, alpha=0.05):
    """The reporting shape Gap 3 asks for: the point estimate computed on the
    real cohort, plus a percentile interval and a bootstrap mean from the
    replicates. The point estimate is deliberately NOT replaced by the
    bootstrap mean — the interval describes the estimate, it does not redefine
    it."""
    lo, hi, n_usable = percentile_ci(replicates, alpha)
    values = np.asarray(replicates, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        'estimate': None if point_estimate is None else round(float(point_estimate), 4),
        'ci_low': None if lo is None else round(lo, 4),
        'ci_high': None if hi is None else round(hi, 4),
        'bootstrap_mean': round(float(values.mean()), 4) if len(values) else None,
        'n_bootstrap_usable': n_usable,
    }


def paired_difference(replicates_a, replicates_b, alpha=0.05):
    """Interval for (model A - model B) taken resample by resample. Returns
    None when fewer than 20 resamples produced a usable number for BOTH models
    (a difference is only defined on draws where both sides exist)."""
    a = np.asarray(replicates_a, dtype=np.float64)
    b = np.asarray(replicates_b, dtype=np.float64)
    both = np.isfinite(a) & np.isfinite(b)
    if both.sum() < 20:
        return None
    diff = a[both] - b[both]
    lo, hi = np.percentile(diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        'mean_difference': round(float(diff.mean()), 4),
        'ci_low': round(float(lo), 4),
        'ci_high': round(float(hi), 4),
        'n_paired': int(both.sum()),
        'excludes_zero': bool(lo > 0 or hi < 0),
    }


def bootstrap_auc_grid(max_time, n_points=BOOTSTRAP_AUC_GRID_POINTS):
    """Coarse evaluation grid for the bootstrap AUC — see the module docstring
    for why the headline per-day grid is not reused here.

    The grid stops strictly BELOW `max_time`. sksurv requires every evaluation
    time to be smaller than the largest observed time in y_test, so a grid whose
    last point equals the maximum follow-up raises for every resample — which
    would show up as an AUC column of all-NaN replicates and an "interval
    unavailable" row rather than as an error. Irrelevant when the horizon binds
    (0-729 d against years of follow-up), but not when a small or heavily
    censored cohort's own maximum is what binds."""
    if max_time is None or max_time <= 1:
        return np.array([])
    upper = float(max_time) * (1.0 - 1e-9)
    if upper <= 1.0:
        return np.array([])
    return np.linspace(1.0, upper, int(n_points))
