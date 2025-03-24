from pkgs.commons import egfr_tv_dynamic_deep_hit_model_path
from pkgs.data.model_data_store import get_train_test_data_egfr, sample
from pkgs.models.dynamicdeephit import DynamicDeepHit
import torch
from torch.utils.data import DataLoader

from pkgs.playground.exp_common import RNNAttentionDataset
from pkgs.playground.exp_common import batch_size, survival_loss, calculate_c_index
from pkgs.experiments.utils import ex_optuna

import os
import numpy as np

ddh_features = ['egfr']
num_risks = 1

def objective(trial):
    df, _ = get_train_test_data_egfr(True)

    dataset = RNNAttentionDataset(df, multiple_risk=False)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = len(ddh_features)
    num_layers = trial.suggest_int("num_layer", 1, 20)
    hidden_dims = [trial.suggest_int(f"hidden_dim_{i}", 16, 256) for i in range(num_layers)]
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    drop_out_lstm = trial.suggest_float('drop_out_rate', 0.1, 0.5)
    drop_out_cause = trial.suggest_float('drop_out_rate', 0.1, 0.5)
    num_epochs = 1

    model = DynamicDeepHit(input_dim, hidden_dims, num_risks, drop_out_lstm, drop_out_cause)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}')
        total_loss = 0
        for features, mask, time_intervals, event_indicators in train_loader:
            optimizer.zero_grad()
            hazard_preds, _ = model(features, mask)
            loss = survival_loss(hazard_preds, time_intervals, event_indicators, num_risks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    c_index = eval_ddh(model, train_loader)

    trial.set_user_attr(key="model", value=model)
    return c_index


def eval_ddh(model, data_loader):
    test_c_indices = []

    with torch.no_grad():
        for features, mask, time_intervals, event_indicators in data_loader:
            hazard_preds, _ = model(features, mask)

            c_indices = calculate_c_index(hazard_preds, time_intervals, event_indicators, num_risks)
            test_c_indices.append(c_indices)

    avg_test_c_indices = np.mean(test_c_indices, axis=0)
    for risk_idx, c_index in enumerate(avg_test_c_indices):
        print(f"Risk {risk_idx + 1} Test C-index: {c_index:.2f}")
    
    return avg_test_c_indices[0] # 1 risk, which is esrd

def run():
    _, df_test = get_train_test_data_egfr(True)

    if os.path.exists(egfr_tv_dynamic_deep_hit_model_path):
        print("Loading from saved weights")
        model = torch.load(egfr_tv_dynamic_deep_hit_model_path, weights_only=False)
    else:
        model = ex_optuna(objective)
        torch.save(model, egfr_tv_dynamic_deep_hit_model_path)
    
    eval_ddh(model, df_test)

    
if __name__ == '__main__':
    run()