import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
import os
import optuna

from pkgs.models.rnnsurv import RNNSurv
from pkgs.data.model_data_store import get_train_test_data_egfr
from pkgs.experiments.utils import evaluate_rnn_model
from pkgs.commons import egfr_tv_rnn_surv_model_path

class RNNSurvDataset(Dataset):
    def __init__(self, df, features, duration_col, event_col):
        self.X = torch.tensor(df[features].values, dtype=torch.float32).unsqueeze(1)
        self.durations = torch.tensor(df[duration_col].values, dtype=torch.float32)
        self.events = torch.tensor(df[event_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.durations[idx], self.events[idx]

def neg_log_partial_likelihood(risk, durations, events):
    risk = risk.view(-1)
    durations_sorted, indices = torch.sort(durations, descending=True)
    risk_sorted = risk[indices]
    events_sorted = events[indices]

    loss = 0.0
    for i in range(len(durations_sorted)):
        event_i = events_sorted[i]
        if event_i == 1:
            risk_set = risk_sorted[i:]
            log_sum_risk = torch.logsumexp(risk_set, dim=0)
            loss -= (risk_sorted[i] - log_sum_risk)
    return loss

def objective(trial):
    features = ['duration_in_days', 'egfr']
    duration_col = 'duration_in_days'
    event_col = 'has_esrd'

    df, _ = get_train_test_data_egfr(True)

    train_dataset = RNNSurvDataset(df, features, duration_col, event_col)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    input_dim = len(features)
    embedding_size = trial.suggest_int('embedding_size', 32, 128)
    num_embedding_layers = trial.suggest_int('num_embedding_layers', 1, 3)
    hidden_dims = trial.suggest_int('hidden_dims', 64, 256)
    num_recurrent_layers = trial.suggest_int('num_recurrent_layers', 1, 3)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    num_epochs = 25

    model = RNNSurv(input_dim, embedding_size, num_embedding_layers, hidden_dims, num_recurrent_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            X_batch, durations_batch, events_batch = batch
            optimizer.zero_grad()
            risk_scores = model(X_batch)
            risk_scores = risk_scores[:, -1, :]
            loss = neg_log_partial_likelihood(risk_scores.squeeze(1), durations_batch, events_batch)
            loss.backward()
            optimizer.step()
    return loss.item()

def train_with_best_params(best_params):
    features = ['duration_in_days', 'egfr']
    duration_col = 'duration_in_days'
    event_col = 'has_esrd'

    df, _ = get_train_test_data_egfr(True)
    train_dataset = RNNSurvDataset(df, features, duration_col, event_col)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    input_dim = len(features)
    embedding_size = best_params['embedding_size']
    num_embedding_layers = best_params['num_embedding_layers']
    hidden_dims = best_params['hidden_dims']
    num_recurrent_layers = best_params['num_recurrent_layers']
    learning_rate = best_params['learning_rate']
    num_epochs = 25

    model = RNNSurv(input_dim, embedding_size, num_embedding_layers, hidden_dims, num_recurrent_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            X_batch, durations_batch, events_batch = batch
            optimizer.zero_grad()
            risk_scores = model(X_batch)
            risk_scores = risk_scores[:, -1, :]
            loss = neg_log_partial_likelihood(risk_scores.squeeze(1), durations_batch, events_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), egfr_tv_rnn_surv_model_path)
    print("Training complete with best parameters.")
    return model

def run():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_params = trial.params
    trained_model = train_with_best_params(best_params)

    _, df_test = get_train_test_data_egfr(True)
    evaluate_rnn_model(trained_model, df_test, ['duration_in_days', 'egfr'])

if __name__ == '__main__':
    run()
