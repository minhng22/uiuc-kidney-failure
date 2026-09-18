"""Shared base class for gbsa/srf/survival_svm: all three are sksurv
estimators (GradientBoostingSurvivalAnalysis/RandomSurvivalForest/
FastSurvivalSVM) with no custom nn.Module of their own (direct calls into
the sksurv library, like pkgs/models/cox.py for lifelines), scored
identically. Each gets its own class (pkgs/models/gbsa.py's GBSAModel,
pkgs/models/srf.py's SRFModel, pkgs/models/survival_svm.py's
SurvivalSVMModel) subclassing this one, rather than three copy-pasted
implementations -- mirrors how pkgs/experiments/gbsa.py/srf.py/
survival_svm.py already share most of their own training/eval logic
structurally.
"""
import numpy as np

from pkgs.data_analysis.types import ExperimentScenario
from pkgs.data_analysis.model_data_store import get_last_observation_data
from pkgs.experiments.utils import get_tv_rnn_model_features


class SksurvModelBase:
    def __init__(self, fitted_model):
        """`fitted_model` is the loaded sksurv estimator (see
        pkgs/experiments/utils.py's load_pkl_and_dill_model -- that's what
        gets dilled to disk from training)."""
        self.fitted_model = fitted_model

    def predictions(self, scenario: ExperimentScenario, split='test'):
        """Fetches get_last_observation_data(scenario)'s flattened
        one-row-per-patient frame itself (matching how these 3 models were
        trained -- see each model's own run_scenario() in
        pkgs/experiments/). `split='train'` scores the training-set
        flattened frame instead (used to fit this model's own Breslow
        baseline hazard).

        `fitted_model.predict(X)` already returns "higher=more risk"
        (sksurv's own convention, e.g. as used directly with sksurv's
        concordance_index_censored in pkgs/experiments/survival_svm.py, no
        inversion) -- matches clinical_validity_analysis.py's convention
        directly. Returns (risk_scores, durations, events, native_prob_fn),
        where native_prob_fn is None for estimators that genuinely have no
        survival function (see _native_prob_fn below)."""
        df_train_flat, df_test_flat = get_last_observation_data(scenario)
        df_flat = df_train_flat if split == 'train' else df_test_flat
        features = get_tv_rnn_model_features(scenario)
        X = df_flat[features].values
        risk_scores = self.fitted_model.predict(X)
        return (risk_scores, df_flat['duration_in_days'].values,
                df_flat['has_esrd'].values, self._native_prob_fn(X))

    def _native_prob_fn(self, X):
        """P(event by horizon_days) read off the fitted estimator's OWN
        survival function, when it has one.

        PAPER_GAPS_EXPERIMENT_PLAN.md Gap 5b asks for each estimator to be
        inspected rather than assumed score-only. Checked against the fitted
        rep99 artifacts:

        - GradientBoostingSurvivalAnalysis was fit with loss='coxph' (the
          sksurv default, and what pkgs/experiments/gbsa.py leaves it at), so
          it carries a Breslow baseline internally and exposes
          `predict_survival_function` -- genuinely native.
        - RandomSurvivalForest exposes `predict_survival_function` (the
          ensemble-averaged Nelson-Aalen estimate per leaf) -- genuinely
          native.
        - FastSurvivalSVM is a pure ranking model: it has no
          `predict_survival_function` and no baseline hazard of any kind, so
          this returns None and the analyzer falls back to its labelled
          risk-score-to-survival conversion, which the report then presents as
          an evaluation of the model PLUS that conversion rather than of the
          SVM's own probabilities.

        sksurv returns one StepFunction per row. They are evaluated lazily and
        cached per horizon: the analyzer asks for ~50 grid points per Brier
        integration, and re-walking n StepFunctions for each is the only part
        of this that is not free.

        Returns None past the last time the estimator's step functions cover
        (`.x.max()`, the training set's largest observed time): beyond it the
        fit carries no information and a flat extrapolation would silently
        understate risk."""
        predict_survival_function = getattr(self.fitted_model, 'predict_survival_function', None)
        if predict_survival_function is None:
            return None
        try:
            step_functions = predict_survival_function(X)
        except Exception:
            return None
        if step_functions is None or len(step_functions) == 0:
            return None
        max_time = float(min(float(np.max(fn.x)) for fn in step_functions))
        cache = {}

        def native_prob_fn(horizon_days):
            if horizon_days < 0 or horizon_days > max_time:
                return None
            key = round(float(horizon_days), 6)
            if key not in cache:
                cache[key] = 1.0 - np.array([float(fn(key)) for fn in step_functions],
                                            dtype=np.float64)
            return cache[key]

        return native_prob_fn
