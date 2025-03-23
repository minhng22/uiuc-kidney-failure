import torch
import torch.optim as optim

from pkgs.models.deepsurv import DeepSurv
from pkgs.data.model_data_store import get_train_test_data_egfr
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
from pkgs.experiments.utils import evaluate_rnn_model, ex_optuna

import os
from pkgs.commons import egfr_ti_deepsurv_model_path

deep_surv_features = ['egfr']

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
    duration_col = 'duration_in_days'
    event_col = 'has_esrd'

    df, _ = get_train_test_data_egfr(False)

    train_dataset = DeepSurvDataset(df, deep_surv_features, duration_col, event_col)
    
    # use full batch to preven neg log likehood loss crash on no positive datapoint in batch
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    input_dim = 1
    num_layers = trial.suggest_int("num_layer", 1, 20)
    hidden_dims = [trial.suggest_int(f"hidden_dim_{i}", 16, 256) for i in range(num_layers)]
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    drop_out = [trial.suggest_float(f"drop_out_rate_{i}", 0.1, 0.5) for i in range(num_layers)]
    num_epochs = 50

    model = DeepSurv(input_dim, hidden_dims, drop_out)
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

    X_test = torch.tensor(df[deep_surv_features].values, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        test_risk_scores = model(X_test)

    c_index = concordance_index(df['duration_in_days'], test_risk_scores, df['has_esrd'])
    trial.set_user_attr(key="model", value=model)
    return c_index

def run():
    _, df_test = get_train_test_data_egfr(False)

    if os.path.exists(egfr_ti_deepsurv_model_path):
        print("Loading from saved weights")
        model = torch.load(egfr_ti_deepsurv_model_path, weights_only=False)
    else:
        model = ex_optuna(objective)
        torch.save(model, egfr_ti_deepsurv_model_path)

    evaluate_rnn_model(model, df_test, deep_surv_features)

if __name__ == '__main__':
    run()

