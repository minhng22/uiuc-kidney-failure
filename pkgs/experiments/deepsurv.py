import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

from pkgs.commons import egfr_ti_deepsurv_model_path
from pkgs.data_analysis.model_data_store import get_train_test_data
from pkgs.data_analysis.types import ExperimentScenario
from pkgs.models.deepsurv import DeepSurv
from pkgs.experiments.utils import c_idx_rnn_model, ex_optuna, round_metric, compute_brier_score_from_risk_scores

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

def get_device():
    return torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def score_model_train(model: DeepSurv, df, features, device):
    X = torch.tensor(df[features].values, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        risk_scores = model(X)
        risk_scores = risk_scores.squeeze()

    c_index = round_metric(concordance_index(df['duration_in_days'], risk_scores.cpu().numpy(), df['has_esrd']))
    print("C-Index Data:", c_index)
    return c_index

def objective(trial):
    device = get_device()
    print(f"Running trial number {trial.number} on device {device}")
    
    duration_col = 'duration_in_days'
    event_col = 'has_esrd'

    df, _ = get_train_test_data(ExperimentScenario.NON_TIME_VARIANT)

    train_dataset = DeepSurvDataset(df, deep_surv_features, duration_col, event_col)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    input_dim = 1 # egfr
    num_layers = trial.suggest_int("num_layer", 1, 20)
    hidden_dims = [trial.suggest_int(f"hidden_dim_{i}", 16, 256) for i in range(num_layers)]
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    drop_out = [trial.suggest_float(f"drop_out_rate_{i}", 0.1, 0.5) for i in range(num_layers)]
    num_epochs = 50

    model = DeepSurv(input_dim, hidden_dims, drop_out).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping parameters
    patience = 5
    best_loss = float('inf')
    patience_counter = 0

    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        total_loss = 0
        for batch in train_loader:
            X_batch, durations_batch, events_batch = [x.to(device) for x in batch]
            optimizer.zero_grad()
            risk_scores = model(X_batch)
            loss = neg_log_partial_likelihood(risk_scores, durations_batch, events_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Check for early stopping
        avg_loss = total_loss / len(train_loader)
        print(f'Average Loss: {avg_loss}')
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'Patience Counter: {patience_counter}')

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    c_index = score_model_train(model, df, deep_surv_features, device)
    
    # Check if current model is better than saved model
    if os.path.exists(egfr_ti_deepsurv_model_path):
        saved_model = torch.load(egfr_ti_deepsurv_model_path, map_location=device)
        saved_c_index = score_model_train(saved_model, df, deep_surv_features, device)
        
        print(f"Current trial C-index: {c_index:.4f}, Previously saved model C-index: {saved_c_index:.4f}")
        if c_index > saved_c_index:
            print("Current model performs better, saving it...")
            torch.save(model, egfr_ti_deepsurv_model_path)
    else:
        print(f"No existing model found, saving current model with C-index: {c_index:.4f}")
        torch.save(model, egfr_ti_deepsurv_model_path)

    trial.set_user_attr(key="model", value=model)
    return c_index

def run():
    device = get_device()
    df, df_test = get_train_test_data(ExperimentScenario.NON_TIME_VARIANT)

    if os.path.exists(egfr_ti_deepsurv_model_path):
        print("Loading from saved weights")
        model = torch.load(egfr_ti_deepsurv_model_path, map_location=device, weights_only=False)
    else:
        model = ex_optuna(objective)
    
    model.to(device)

    # Compute C-Index on test data
    X_test = torch.tensor(df_test[deep_surv_features].values, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        test_risk_scores = model(X_test).squeeze().cpu().numpy()

    c_index = round_metric(concordance_index(df_test['duration_in_days'], test_risk_scores, df_test['has_esrd']))
    print("C-Index on Test Data:", c_index)

    # Compute Brier Score
    brier_score = compute_brier_score_from_risk_scores(df, df_test, test_risk_scores)
    if brier_score is not None:
        print(f'Integrated Brier Score Test: {brier_score}')

    # Compute time-dependent AUC
    times = np.arange(1, 365, 1)
    risk_scores = test_risk_scores

    y_train = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df)
    y_test = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df_test)

    print(f'Risk scores shape: {risk_scores.shape}')
    print(f'First 10 risk scores: {risk_scores[:10]}')
    print(f'y_train shape: {y_train.shape}')
    print(f'y_test shape: {y_test.shape}')

    _, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)
    print(f'Mean AUC: {round_metric(mean_auc)}')

if __name__ == '__main__':
    run()

