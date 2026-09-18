"""Cox (lifelines CoxTimeVaryingFitter/CoxPHFitter) has no custom nn.Module --
it's a direct call into the lifelines library, so unlike deepsurv/
dynamicdeephit/hazard_transformer/rnnsurv there's no architecture class
already sitting in this package. CoxModel below gives it the same "model
layer" home those four have -- a class holding the fitted estimator, with a
predictions() method on it, same shape as
HazardTransformer.predictions()/DynamicDeepHit.predictions()/etc.
"""
import numpy as np

from pkgs.data_analysis.types import ExperimentScenario
from pkgs.data_analysis.model_data_store import get_last_observation_data


class CoxModel:
    def __init__(self, fitted_model):
        """`fitted_model` is the loaded lifelines CoxTimeVaryingFitter/
        CoxPHFitter (see pkgs/experiments/utils.py's load_pkl_and_dill_model
        -- that's what gets dilled to disk from training, so callers wrap
        it in CoxModel right after loading, not before)."""
        self.fitted_model = fitted_model

    def predictions(self, scenario: ExperimentScenario, split='test'):
        """Fetches get_last_observation_data(scenario)'s flattened
        one-row-per-patient frame itself (each patient's LAST observation)
        rather than every raw start/stop interval row of the time-varying
        data. `predict_partial_hazard` only needs the covariate columns
        (partial hazard = exp(beta^T x), no time-interval dependency), so
        this works directly on the flattened frame -- and matches how ddh/
        hazard_transformer/logistic_hazard (fetch their own multi-row
        time-varying frame instead) and deepsurv/gbsa/srf/survival_svm/
        weibul (already use this same flattened frame) are evaluated.
        Before Stage 2.2's fix, cox/rnn_surv/kfre were the only 3 of 11
        models still scored per lab-event ROW, which massively dilutes the
        row-level event rate vs. the true patient-level rate (a patient who
        eventually has the event still has dozens-to-hundreds of earlier
        rows correctly labeled "no event yet") -- see
        generated_data/rep99/stage2_2_debug_report.txt for the concrete
        numbers (twenty_features_heterogeneous: 50% patient-level
        ESRD-positive rate vs. 3.25% row-level event rate) and why this was
        masking real (lack of) discrimination behind a deceptively low
        Brier score. `split='train'` scores the training-set flattened
        frame instead (used to fit this model's own Breslow baseline
        hazard). Returns (risk_scores, durations, events, native_prob_fn)."""
        df_train_flat, df_test_flat = get_last_observation_data(scenario)
        df_flat = df_train_flat if split == 'train' else df_test_flat
        risk_scores = self.fitted_model.predict_partial_hazard(df_flat).values.flatten()
        native_prob_fn = self._native_prob_fn(risk_scores)
        return risk_scores, df_flat['duration_in_days'].values, df_flat['has_esrd'].values, native_prob_fn

    def _native_prob_fn(self, risk_scores):
        """P(event by horizon_days) from the FITTED model's OWN baseline
        cumulative hazard: S(t|x) = exp(-H0(t) * partial_hazard(x)), i.e.
        `baseline_cumulative_hazard_` as lifelines estimated it during fit(),
        combined with this patient's partial hazard.

        This replaces returning None (PAPER_GAPS_EXPERIMENT_PLAN.md Gap 5b:
        "current wrappers return `None` for native probabilities. This does not
        prove the underlying estimator lacks a survival function"). The
        difference from the analyzer's generic fallback is which baseline is
        used: the fallback re-fits a Breslow baseline from this model's risk
        SCORES on the training set, treating the model as an opaque ranker,
        whereas this reads the baseline the Cox fit itself produced.

        CoxTimeVaryingFitter has no `predict_survival_function` (verified
        against the fitted rep99 artifacts, which expose
        `baseline_cumulative_hazard_`/`baseline_survival_`/
        `predict_partial_hazard` and nothing else), because a start/stop model
        has no single covariate vector per subject to carry forward. The
        carried-forward vector used here is each patient's LAST observation --
        the same row this model is scored on and the same landmark every other
        model in the comparison uses -- so the curve is "this patient's risk
        from the landmark onward, holding their most recent labs fixed". That
        last-observation-carried-forward step is an assumption of the reading,
        not of the fit, and the analyzer labels the resulting metric
        accordingly.

        Returns None past the last time the baseline hazard covers: beyond that
        the fit carries no information, and extrapolating a flat H0 would
        silently understate risk."""
        bch = getattr(self.fitted_model, 'baseline_cumulative_hazard_', None)
        if bch is None or len(bch) == 0:
            return None
        times = np.asarray(bch.index, dtype=np.float64)
        H0 = np.asarray(bch.iloc[:, 0], dtype=np.float64)
        max_time = float(times.max())
        partial_hazard = np.asarray(risk_scores, dtype=np.float64)

        def native_prob_fn(horizon_days):
            if horizon_days < 0 or horizon_days > max_time:
                return None
            idx = int(np.searchsorted(times, float(horizon_days), side='right')) - 1
            H0_t = H0[idx] if idx >= 0 else 0.0
            return 1.0 - np.exp(-partial_hazard * H0_t)

        return native_prob_fn
