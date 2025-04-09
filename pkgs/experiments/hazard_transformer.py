from pkgs.commons import egfr_tv_hazard_transformer_model_path
from pkgs.data.model_data_store import get_train_test_data
from pkgs.models.hazard_transformer import HazardTransformer
import torch
from torch.utils.data import DataLoader
from pkgs.playground.exp_common import batch_size, RNNAttentionDataset, calculate_c_index, survival_loss
import numpy as np
import os
from pkgs.experiments.utils import ex_optuna, get_tv_rnn_model_features
from pkgs.data.types import ExperimentScenario

num_risks = 1

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial, scenario_name: ExperimentScenario):
    device = get_device()

    print(f"Running trial {trial.number} for {scenario_name} on device {device}")
    df, _ = get_train_test_data(scenario_name)

    dataset = RNNAttentionDataset(df, scenario_name)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = len(get_tv_rnn_model_features(scenario_name))
    num_layers = trial.suggest_int("num_layers", 2, 64)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    drop_out = trial.suggest_float('drop_out_rate', 0.1, 0.5)
    num_epochs = 50
    nhead = trial.suggest_int("n_head", 1, 8)
    nhead_factor = trial.suggest_int("nhead_factor", 1, 16)
    hidden_dims = nhead * nhead_factor
    max_time = trial.suggest_int("max_time", 50, 200)

    model = HazardTransformer(input_dim, hidden_dims, num_risks, num_layers, nhead, drop_out, max_time=max_time).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _ in range(num_epochs):
        for features, mask, time_intervals, event_indicators in train_loader:
            features, mask, time_intervals, event_indicators = [x.to(device) for x in (features, mask, time_intervals, event_indicators)]
            optimizer.zero_grad()
            
            eval_times = torch.linspace(0, model.max_time, 100).to(device)
            eval_times = eval_times.unsqueeze(0).repeat(features.size(0), 1)
            
            hazard_preds, _, _ = model(features, mask, eval_times)
            loss = survival_loss(hazard_preds, time_intervals, event_indicators, num_risks)
            loss.backward()
            optimizer.step()

    c_index = eval_ht(model, train_loader, device)
    trial.set_user_attr(key="model", value=model)
    return c_index

def eval_ht(model: HazardTransformer, data_loader, device):
    test_c_indices = []

    model.eval()
    with torch.no_grad():
        for features, mask, time_intervals, event_indicators in data_loader:
            features, mask = features.to(device), mask.to(device)
            eval_times = torch.linspace(0, model.max_time, 100).to(device)
            eval_times = eval_times.unsqueeze(0).repeat(features.size(0), 1)
            
            hazard_preds, _, _ = model(features, mask, eval_times)
            c_indices = calculate_c_index(hazard_preds, time_intervals, event_indicators, num_risks)
            test_c_indices.append(c_indices)

    avg_test_c_indices = np.mean(test_c_indices, axis=0)
    for risk_idx, c_index in enumerate(avg_test_c_indices):
        print(f"Risk {risk_idx + 1} Test C-index: {c_index:.2f}")
    
    return avg_test_c_indices[0]  # Return c-index for the single risk

def run(scenario_name: ExperimentScenario):
    device = get_device()
    _, df_test = get_train_test_data(ExperimentScenario.TIME_VARIANT)
    if os.path.exists(egfr_tv_hazard_transformer_model_path):
        print("Loading from saved weights")
        model = torch.load(egfr_tv_hazard_transformer_model_path, map_location=device)
    else:
        model = ex_optuna(lambda trial: objective(trial, scenario_name))
        torch.save(model, egfr_tv_hazard_transformer_model_path)
    
    model.to(device)

    eval_ht(model, df_test, device)

if __name__ == '__main__':
    run(ExperimentScenario.TIME_VARIANT)
    run(ExperimentScenario.HETEROGENEOUS)
    run(ExperimentScenario.EGFR_COMPONENTS)