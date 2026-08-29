import torch
import torch.nn as nn

from pkgs.data_analysis.types import ExperimentScenario
from pkgs.data_analysis.model_data_store import get_last_observation_data
from pkgs.experiments.utils import get_tv_rnn_model_features

class RNNSurv(nn.Module):
    """RNN-Surv with a discrete event-time PMF output.

    The old head independently sigmoid-activated each time interval and
    called the results survival probabilities.  They were neither a valid
    distribution nor guaranteed to form a monotonic survival curve.  This
    head instead softmax-normalizes over time: its cumulative sum is the CIF
    and one minus that cumulative sum is survival.

    Models saved under the old sigmoid-head architecture must be retrained;
    applying a softmax to their weights would not make them valid PMF models.
    """

    ARCHITECTURE_VERSION = 2

    def __init__(self, input_size, embedding_size, num_embedding_layers, hidden_size,
                 num_recurrent_layers, num_time_intervals, max_time=730.0):
        """
        Args:
            input_size (int): Number of input features (including time interval identifier).
            embedding_size (int): Dimensionality of the embedding layers.
            num_embedding_layers (int): Number of embedding layers (N1).
            hidden_size (int): Size of the LSTM hidden state.
            num_recurrent_layers (int): Number of LSTM layers (N2).
            num_time_intervals (int): Number of discrete event-time intervals (K).
            max_time (float): End of the training follow-up horizon in days.
        """
        super(RNNSurv, self).__init__()
        # Keep this on the instance (not only the class) so old pickled
        # sigmoid-head models can be reliably detected before inference.
        self.architecture_version = self.ARCHITECTURE_VERSION
        self.embedding_layers = nn.Sequential()
        for i in range(num_embedding_layers):
            input_dim = input_size if i == 0 else embedding_size
            self.embedding_layers.add_module(f'embedding_{i}', nn.Linear(input_dim, embedding_size))
            if i < num_embedding_layers - 1:
                self.embedding_layers.add_module(f'relu_embedding_{i}', nn.ReLU())

        self.rnn = nn.LSTM(embedding_size, hidden_size, num_recurrent_layers, batch_first=True)
        
        self.num_time_intervals = num_time_intervals
        self.max_time = float(max_time)

        # One logit per event-time interval.  Softmax in forward() makes this
        # a PMF rather than K independent sigmoid values.
        self.output_layer = nn.Linear(hidden_size, num_time_intervals)

    def forward(self, x):
        # Pass the input through the embedding layers
        embedded = self.embedding_layers(x)

        # The recurrent layers (LSTM)
        out, _ = self.rnn(embedded)

        event_pmf = torch.softmax(self.output_layer(out), dim=-1)

        # Scalar risk is expected event earliness: high when PMF mass is in
        # early intervals, low for later event times.
        interval_earliness = torch.linspace(
            1.0, 0.0, self.num_time_intervals, device=x.device,
            dtype=event_pmf.dtype,
        )
        risk_scores = (event_pmf[:, -1, :] * interval_earliness).sum(dim=1, keepdim=True)

        return event_pmf, risk_scores

    def predictions(self, scenario: ExperimentScenario, split='test'):
        """Used by pkgs/data_analysis/clinical_validity_analysis.py's Stage
        2.1/2.2 calibration + decision-curve report. Fetches
        get_last_observation_data(scenario)'s flattened one-row-per-patient
        frame itself (each patient's last observation, as a length-1
        sequence) rather than every raw lab-event row -- see
        pkgs/models/cox.py's CoxModel.predictions docstring for why
        (cox/rnn_surv/kfre were the last 3 of 11 models still scored per
        row until Stage 2.2's fix). `split='train'` scores the
        training-set flattened frame instead (used to fit this model's own
        Breslow baseline hazard).

        Returns RNN-Surv's PMF-derived risk and native horizon
        probabilities. Returns (risk_scores, durations, events, native_prob_fn)."""
        if getattr(self, 'architecture_version', 1) != 2:
            raise ValueError(
                "This RNN-Surv checkpoint uses the obsolete sigmoid head; retrain it before analysis."
            )
        df_train_flat, df_test_flat = get_last_observation_data(scenario)
        df_flat = df_train_flat if split == 'train' else df_test_flat
        durations = df_flat['duration_in_days'].values
        events = df_flat['has_esrd'].values
        features = get_tv_rnn_model_features(scenario)
        X = torch.tensor(df_flat[features].values, dtype=torch.float32).unsqueeze(1)

        self.eval()
        with torch.no_grad():
            event_pmf, risk_scores_t = self(X)
            risk_scores_t = risk_scores_t.squeeze()
        risk_scores = risk_scores_t.cpu().numpy()
        cif = event_pmf[:, -1, :].cumsum(dim=1).cpu().numpy()
        max_bin = cif.shape[1] - 1
        max_time = float(getattr(self, 'max_time', 730.0))

        def native_prob_fn(horizon_days):
            if horizon_days < 0 or horizon_days > max_time:
                return None
            bin_idx = int(round(horizon_days / max_time * max_bin))
            return cif[:, min(max(bin_idx, 0), max_bin)]

        return risk_scores, durations, events, native_prob_fn
