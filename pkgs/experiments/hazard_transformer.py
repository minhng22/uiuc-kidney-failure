from pkgs.commons import egfr_tv_hazard_transformer_model_path
from pkgs.data.model_data_store import get_train_test_data_egfr, mini
from pkgs.models.hazard_transformer import HazardTransformer
import torch
from torch.utils.data import DataLoader
from pkgs.playground.exp_common import batch_size, time_bins, RNNAttentionDataset, calculate_c_index, survival_loss
import numpy as np
import os
from pkgs.experiments.utils import ex_optuna

num_risks = 1


def objective(trial):
    df, df_test = get_train_test_data_egfr(True)
    df = mini(df)

    dataset = RNNAttentionDataset(df, multiple_risk=False)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = 2
    n_layer = trial.suggest_int("hidden_dim", 2, 64)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    drop_out = trial.suggest_float('drop_out_rate', 0.1, 0.5)
    num_epochs = 1
    nhead = trial.suggest_int("n_head", 1, 8)
    nhead_factor = trial.suggest_int("nhead_factor", 1, 16)
    hidden_dims = nhead * nhead_factor

    model = HazardTransformer(input_dim, hidden_dims, time_bins, num_risks, n_layer, nhead, drop_out)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for features, mask, time_intervals, event_indicators in train_loader:
            optimizer.zero_grad()
            hazard_preds, _ = model(features, mask)
            loss = survival_loss(hazard_preds, time_intervals, event_indicators, num_risks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    c_index = eval_ht(model, df_test)

    trial.set_user_attr(key="model", value=model)
    return c_index


def eval_ht(model, df_test):
    test_c_indices = []

    test_dataset = RNNAttentionDataset(df_test, multiple_risk=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for features, mask, time_intervals, event_indicators in test_dataloader:
            hazard_preds, _ = model(features, mask)
            c_indices = calculate_c_index(hazard_preds, time_intervals, event_indicators, num_risks)
            test_c_indices.append(c_indices)

    avg_test_c_indices = np.mean(test_c_indices, axis=0)
    for risk_idx, c_index in enumerate(avg_test_c_indices):
        print(f"Risk {risk_idx + 1} Test C-index: {c_index:.2f}")
    
    return avg_test_c_indices[0] # 1 risk, which is esrd


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
