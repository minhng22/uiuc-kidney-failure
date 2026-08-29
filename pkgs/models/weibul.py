"""WeibullAFTFitter (lifelines) has no custom nn.Module of its own -- a
direct call into the lifelines library, like pkgs/models/cox.py. WeibulModel
below gives it the same "model layer" home the neural-net models have.
"""
import numpy as np

from pkgs.data_analysis.types import ExperimentScenario
from pkgs.data_analysis.model_data_store import get_last_observation_data
from pkgs.experiments.utils import get_tv_rnn_model_features


class WeibulModel:
    def __init__(self, fitted_model):
        """`fitted_model` is the loaded lifelines WeibullAFTFitter (see
        pkgs/experiments/utils.py's load_pkl_and_dill_model -- that's what
        gets dilled to disk from training)."""
        self.fitted_model = fitted_model

    def predictions(self, scenario: ExperimentScenario, split='test'):
        """Fetches get_last_observation_data(scenario)'s flattened
        one-row-per-patient frame itself, restricted to this scenario's
        feature columns plus duration_in_days/has_esrd (see
        pkgs/experiments/weibul.py's run_scenario() for why
        duration_in_days==0 needs a small positive floor -- lifelines
        errors on duration==0). `split='train'` scores the training-set
        flattened frame instead (used to fit this model's own Breslow
        baseline hazard).

        WeibullAFTFitter predicts a median SURVIVAL TIME, not a risk
        score -- "higher=longer survival", the opposite of
        clinical_validity_analysis.py's convention. Negate it (same
        transform pkgs/experiments/weibul.py's own Brier-score call already
        uses: `-predicted_survival_times`) to get "higher=riskier". Returns
        (risk_scores, durations, events, None) -- this model has no native
        per-horizon output."""
        df_train_flat, df_test_flat = get_last_observation_data(scenario)
        df_flat = df_train_flat if split == 'train' else df_test_flat
        features = get_tv_rnn_model_features(scenario)
        cols = features + ['duration_in_days', 'has_esrd']
        df_flat_selected = df_flat[cols].copy()
        df_flat_selected['duration_in_days'] = df_flat_selected['duration_in_days'].replace(0, 1e-5)

        predicted_survival_times = self.fitted_model.predict_median(df_flat_selected)
        risk_scores = (-predicted_survival_times.values if hasattr(predicted_survival_times, 'values')
                       else -np.asarray(predicted_survival_times))
        return risk_scores, df_flat_selected['duration_in_days'].values, df_flat_selected['has_esrd'].values, None
