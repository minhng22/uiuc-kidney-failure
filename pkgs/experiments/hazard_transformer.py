from pkgs.commons import egfr_tv_hazard_transformer_model_path
from pkgs.data.model_data_store import get_train_test_data_egfr, mini
from pkgs.models.hazard_transformer import HazardTransformer
import torch
from torch.utils.data import DataLoader
from pkgs.playground.exp_common import batch_size, RNNAttentionDataset, calculate_c_index, survival_loss
import numpy as np
import os
from pkgs.experiments.utils import ex_optuna

num_risks = 1

def objective(trial):
    df, _ = get_train_test_data_egfr(True)

    dataset = RNNAttentionDataset(df, multiple_risk=False)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = 2
    num_layers = trial.suggest_int("num_layers", 2, 64)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    drop_out = trial.suggest_float('drop_out_rate', 0.1, 0.5)
    num_epochs = 3
    nhead = trial.suggest_int("n_head", 1, 8)
    nhead_factor = trial.suggest_int("nhead_factor", 1, 16)
    hidden_dims = nhead * nhead_factor
    max_time = trial.suggest_int("max_time", 50, 200)

    model = HazardTransformer(input_dim, hidden_dims, num_risks, num_layers, nhead, drop_out, max_time=max_time)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _ in range(num_epochs):
        for features, mask, time_intervals, event_indicators in train_loader:
            optimizer.zero_grad()
            
            eval_times = torch.linspace(0, model.max_time, 100)
            eval_times = eval_times.unsqueeze(0).repeat(features.size(0), 1)
            
            hazard_preds, _, _ = model(features, mask, eval_times)
            loss = survival_loss(hazard_preds, time_intervals, event_indicators, num_risks)
            loss.backward()
            optimizer.step()

    c_index = eval_ht(model, df)
    trial.set_user_attr(key="model", value=model)
    return c_index

def eval_ht(model, df):
    test_c_indices = []
    test_dataset = RNNAttentionDataset(df, multiple_risk=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        for features, mask, time_intervals, event_indicators in test_dataloader:
            eval_times = torch.linspace(0, model.max_time, 100)
            eval_times = eval_times.unsqueeze(0).repeat(features.size(0), 1)
            
            hazard_preds, _, _ = model(features, mask, eval_times)
            c_indices = calculate_c_index(hazard_preds, time_intervals, event_indicators, num_risks)
            test_c_indices.append(c_indices)

    avg_test_c_indices = np.mean(test_c_indices, axis=0)
    for risk_idx, c_index in enumerate(avg_test_c_indices):
        print(f"Risk {risk_idx + 1} Test C-index: {c_index:.2f}")
    
    return avg_test_c_indices[0]  # Return c-index for the single risk

def run():
    _, df_test = get_train_test_data_egfr(True)
    if os.path.exists(egfr_tv_hazard_transformer_model_path):
        print("Loading from saved weights")
        model = torch.load(egfr_tv_hazard_transformer_model_path, weights_only=False)
    else:
        model = ex_optuna(objective)
        torch.save(model, egfr_tv_hazard_transformer_model_path)
    
    eval_ht(model, df_test)

if __name__ == '__main__':
    run()