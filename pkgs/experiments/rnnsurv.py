import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index

from pkgs.models.rnnsurv import RNNSurv
from pkgs.data.model_data_store import get_train_test_data
from pkgs.experiments.utils import round_metric, ex_optuna, get_tv_rnn_model_features
from pkgs.commons import egfr_tv_rnn_surv_model_path, hg_rnn_surv_model_path, egfr_components_rnn_surv_model_path
from pkgs.data.types import ExperimentScenario
from sksurv.metrics import cumulative_dynamic_auc
import numpy as np

import os
from sksurv.util import Surv

class RNNSurvDataset(Dataset):
    def __init__(self, df, features, duration_col, event_col):
        self.X = torch.tensor(df[features].values, dtype=torch.float32).unsqueeze(1)
        self.durations = torch.tensor(df[duration_col].values, dtype=torch.float32)
        self.events = torch.tensor(df[event_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.durations[idx], self.events[idx]

def rnn_surv_loss(survival_probabilities, risk_scores, durations, events, time_intervals, cross_entropy_loss_weight):
    """
    Loss function based on the original RNNSurv paper.

    Args:
        survival_probabilities (torch.Tensor): Predicted survival probabilities
            of shape (batch, seq_len, num_time_intervals).
        risk_scores (torch.Tensor): Predicted risk scores of shape (batch, 1).
        durations (torch.Tensor): True durations of shape (batch,).
        events (torch.Tensor): Event indicators of shape (batch,).
        time_intervals (torch.Tensor): Tensor of time interval endpoints.

    Returns:
        torch.Tensor: The combined loss value.
    """
    n_patients = survival_probabilities.size(0)
    n_time_intervals = survival_probabilities.size(2)
    max_observed_time = time_intervals[-1]

    loss_1 = 0.0
    for i in range(n_patients):
        observed_time = durations[i]
        event = events[i]
        for k in range(n_time_intervals):
            t_start = 0 if k == 0 else time_intervals[k-1]
            t_end = time_intervals[k]
            if t_start <= observed_time < t_end:
                indicator = 1.0
                survival_prob = survival_probabilities[i, -1, k] # Use last time step
                term = (event * torch.log(torch.clamp(1 - survival_prob, 1e-7, 1.0)) +
                        (1 - event) * torch.log(torch.clamp(survival_prob, 1e-7, 1.0)))
                loss_1 -= indicator * term
                break 

    loss_1 /= n_patients

    loss_2 = 0.0
    for i in range(n_patients):
        observed_time = durations[i]
        survival_prob_at_observed_time = torch.tensor(1.0)
        for k in range(n_time_intervals):
            t_start = 0 if k == 0 else time_intervals[k-1]
            t_end = time_intervals[k]
            if observed_time < t_end:
                survival_prob_at_observed_time = survival_probabilities[i, -1, k] # Use last time step
                break
            elif observed_time >= max_observed_time:
                survival_prob_at_observed_time = survival_probabilities[i, -1, -1] # Last interval
                break

        loss_2 += (risk_scores[i] - (-torch.log(torch.clamp(survival_prob_at_observed_time, 1e-7, 1.0))))**2

    loss_2 /= n_patients

    total_loss = cross_entropy_loss_weight * loss_1 + (1 - cross_entropy_loss_weight) * loss_2
    return total_loss

def objective(trial, scenario_name: ExperimentScenario):
    device = get_device()

    print(f"Running trial number {trial.number} for {scenario_name} on device {device}")
    duration_col = 'duration_in_days'
    event_col = 'has_esrd'
    num_time_intervals = trial.suggest_int('num_time_intervals', 10, 50)
    rnn_surv_features = get_tv_rnn_model_features(scenario_name)

    df, _ = get_train_test_data(scenario_name)

    train_dataset = RNNSurvDataset(df, rnn_surv_features, duration_col, event_col)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    input_dim = len(rnn_surv_features)
    embedding_size = trial.suggest_int('embedding_size', 32, 128)
    num_embedding_layers = trial.suggest_int('num_embedding_layers', 1, 3)
    hidden_dims = trial.suggest_int('hidden_dims', 64, 256)
    num_recurrent_layers = trial.suggest_int('num_recurrent_layers', 1, 3)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    cross_entropy_loss_weight = trial.suggest_float('cross_entropy_loss_weight', 0.1, 0.9)

    num_epochs = 50

    # Define time intervals based on the training data
    max_duration = df[duration_col].max()
    time_intervals = torch.linspace(0, max_duration, num_time_intervals + 1)[1:].to(device)

    model = RNNSurv(input_dim, embedding_size, num_embedding_layers, hidden_dims, num_recurrent_layers, num_time_intervals).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _ in range(num_epochs):
        print(f"Epoch {_ + 1}/{num_epochs}")
        for batch in train_loader:
            X_batch, durations_batch, events_batch = [x.to(device) for x in batch]
            optimizer.zero_grad()
            survival_probabilities, risk_scores = model(X_batch)
            loss = rnn_surv_loss(survival_probabilities, risk_scores, durations_batch, events_batch, time_intervals, cross_entropy_loss_weight)
            loss.backward()
            optimizer.step()

    trial.set_user_attr(key="model", value=model)
    return score_model_train(model, df, rnn_surv_features, device)

def score_model_train(model: RNNSurv, df, features, device):
    X_test = torch.tensor(df[features].values, dtype=torch.float32).unsqueeze(1).to(device)
    model.eval()
    with torch.no_grad():
        _, test_risk_scores = model(X_test)
        test_risk_scores = test_risk_scores.squeeze()

    c_index = round_metric(concordance_index(df['duration_in_days'], test_risk_scores.cpu().numpy(), df['has_esrd']))
    print("C-Index on Test Data:", c_index)

    return c_index

def get_device():
    return torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# Update the run function to use the device
def run(scenario_name: ExperimentScenario):
    device = get_device()
    df, df_test = get_train_test_data(scenario_name)

    model_path_dict = {
        ExperimentScenario.TIME_VARIANT: egfr_tv_rnn_surv_model_path,
        ExperimentScenario.HETEROGENEOUS: hg_rnn_surv_model_path,
        ExperimentScenario.EGFR_COMPONENTS: egfr_components_rnn_surv_model_path
    }
    model_saved_path = model_path_dict[scenario_name]

    if os.path.exists(model_saved_path):
        print("Loading from saved weights")
        model = torch.load(model_saved_path, map_location=device, weights_only=False)
    else:
        model = ex_optuna(lambda trial: objective(trial, scenario_name))
        torch.save(model, model_saved_path)

    model.to(device)

    X_test = torch.tensor(df_test[get_tv_rnn_model_features(scenario_name)].values, dtype=torch.float32).unsqueeze(1).to(device)
    model.eval()
    with torch.no_grad():
        _, test_risk_scores = model(X_test)
        test_risk_scores = test_risk_scores.squeeze()

    c_index = round_metric(concordance_index(df_test['duration_in_days'], test_risk_scores.cpu().numpy(), df_test['has_esrd']))
    print("C-Index on Test Data:", c_index)

    times = np.arange(1, 365, 1)
    y_train = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df)
    y_test = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df_test)
    _, mean_auc = cumulative_dynamic_auc(y_train, y_test, test_risk_scores.cpu().numpy(), times)

    print(f"Mean time-dependent AUC: {mean_auc:.4f}")

if __name__ == '__main__':
    run(ExperimentScenario.TIME_VARIANT)
    run(ExperimentScenario.HETEROGENEOUS)
    run(ExperimentScenario.EGFR_COMPONENTS)