import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from pkgs.models.deepsurv import DeepSurv
from pkgs.data.model_data_store import get_train_test_data_egfr
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
from pkgs.experiments.utils import report_metric

import os
from pkgs.commons import egfr_ti_deepsurv_model_path

class DeepSurvDataset(Dataset):
    def __init__(self, df, features, duration_col, event_col):
        self.X = torch.tensor(df[features].values, dtype=torch.float32)
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
    features = ['egfr']
    duration_col = 'duration_in_days'
    event_col = 'has_esrd'

    df, df_test = get_train_test_data_egfr(False)

    train_dataset = DeepSurvDataset(df, features, duration_col, event_col)
    
    # use full batch to preven neg log likehood loss crash on no positive datapoint in batch
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    input_dim = 1
    hidden_dims = [trial.suggest_int(f"hidden_dim_{i}", 16, 256) for i in range(10)]
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    num_epochs = 25

    model = DeepSurv(input_dim, hidden_dims)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _ in range(num_epochs):
        for batch in train_loader:
            X_batch, durations_batch, events_batch = batch
            optimizer.zero_grad()
            risk_scores = model(X_batch)
            loss = neg_log_partial_likelihood(risk_scores, durations_batch, events_batch)
            loss.backward()
            optimizer.step()

    X_test = torch.tensor(df_test[features].values, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        test_risk_scores = model(X_test)

    c_index = concordance_index(df_test['duration_in_days'], test_risk_scores, df_test['has_esrd'])
    return c_index

def run():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    features = ['egfr']
    duration_col = 'duration_in_days'
    event_col = 'has_esrd'

    df, df_test = get_train_test_data_egfr(False)

    train_dataset = DeepSurvDataset(df, features, duration_col, event_col)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    input_dim = 1
    hidden_dims = [trial.params[f"hidden_dim_{i}"] for i in range(5)]
    learning_rate = trial.params['learning_rate']
    num_epochs = 25

    model = DeepSurv(input_dim, hidden_dims)

    if os.path.exists(egfr_ti_deepsurv_model_path):
        print("Loading from saved weights")
        model.load_state_dict(torch.load(egfr_ti_deepsurv_model_path, weights_only=True))
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(num_epochs):
            for batch in train_loader:
                X_batch, durations_batch, events_batch = batch
                optimizer.zero_grad()
                risk_scores = model(X_batch)
                loss = neg_log_partial_likelihood(risk_scores, durations_batch, events_batch)
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), egfr_ti_deepsurv_model_path)
        print("Training complete.")

    X_test = torch.tensor(df_test[features].values, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        test_risk_scores = model(X_test)

    c_index = report_metric(concordance_index(df_test['duration_in_days'], test_risk_scores, df_test['has_esrd']))
    print("C-Index on Test Data:", c_index)

if __name__ == '__main__':
    run()

