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
        directly. Returns (risk_scores, durations, events, None) -- these 3
        models have no native per-horizon output."""
        df_train_flat, df_test_flat = get_last_observation_data(scenario)
        df_flat = df_train_flat if split == 'train' else df_test_flat
        features = get_tv_rnn_model_features(scenario)
        X = df_flat[features].values
        risk_scores = self.fitted_model.predict(X)
        return risk_scores, df_flat['duration_in_days'].values, df_flat['has_esrd'].values, None
