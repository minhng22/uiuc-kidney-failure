"""Cox (lifelines CoxTimeVaryingFitter/CoxPHFitter) has no custom nn.Module --
it's a direct call into the lifelines library, so unlike deepsurv/
dynamicdeephit/hazard_transformer/rnnsurv there's no architecture class
already sitting in this package. CoxModel below gives it the same "model
layer" home those four have -- a class holding the fitted estimator, with a
predictions() method on it, same shape as
HazardTransformer.predictions()/DynamicDeepHit.predictions()/etc.
"""
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
        # A native per-horizon survival probability (S_0(t)^partial_hazard, via
        # CoxTimeVaryingFitter's baseline cumulative hazard) exists in principle,
        # but is nontrivial to get right for start/stop interval data — left as
        # a documented approximation (same one already used for Brier score
        # elsewhere in this codebase) rather than a real fix.
        return risk_scores, df_flat['duration_in_days'].values, df_flat['has_esrd'].values, None
