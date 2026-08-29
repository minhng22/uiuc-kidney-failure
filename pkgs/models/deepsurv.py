import torch
import torch.nn as nn
import torch.optim as optim

from pkgs.data_analysis.types import ExperimentScenario
from pkgs.data_analysis.model_data_store import get_last_observation_data
from pkgs.experiments.utils import get_tv_rnn_model_features

class DeepSurv(nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_outs):
        super(DeepSurv, self).__init__()
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_outs[i]))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predictions(self, scenario: ExperimentScenario, split='test'):
        """Used by pkgs/data_analysis/clinical_validity_analysis.py's Stage
        2.1/2.2 calibration + decision-curve report. Fetches
        get_last_observation_data(scenario)'s flattened one-row-per-patient
        frame itself (matching how this model was trained -- see
        pkgs/experiments/deepsurv.py's run_scenario()). `split='train'`
        scores the training-set flattened frame instead (used only to fit
        this model's own Breslow baseline hazard).

        DeepSurv's raw output IS already "higher=more hazard=shorter
        survival" by construction (Cox partial-likelihood training, see
        pkgs/experiments/deepsurv.py's neg_log_partial_likelihood) --
        matches clinical_validity_analysis.py's uniform "higher=riskier"
        convention directly, no transform needed. Returns
        (risk_scores, durations, events, None) -- this model has no native
        per-horizon output."""
        df_train_flat, df_test_flat = get_last_observation_data(scenario)
        df_flat = df_train_flat if split == 'train' else df_test_flat
        features = get_tv_rnn_model_features(scenario)
        X = torch.tensor(df_flat[features].values, dtype=torch.float32)

        self.eval()
        with torch.no_grad():
            risk_scores = self(X).squeeze().cpu().numpy()
        return risk_scores, df_flat['duration_in_days'].values, df_flat['has_esrd'].values, None
