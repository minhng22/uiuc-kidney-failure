from pkgs.commons import egfr_tv_dynamic_deep_hit_model_path, hg_dynamic_deep_hit_model_path, egfr_components_dynamic_deep_hit_model_path
from pkgs.data.model_data_store import get_train_test_data
from pkgs.models.dynamicdeephit import DynamicDeepHit
import torch
from torch.utils.data import DataLoader

from pkgs.experiments.utils import RNNAttentionDataset
from pkgs.experiments.utils import ex_optuna, get_tv_rnn_model_features, calculate_c_index, combine_loss
from pkgs.data.types import ExperimentScenario

import os
import numpy as np
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc


num_risks = 1 # esrd

def objective(trial, scenario_name: ExperimentScenario):
    device = get_device()

    print(f"Running trial {trial.number} for {scenario_name} on device {device}")
    df, _ = get_train_test_data(scenario_name)

    dataset = RNNAttentionDataset(df, scenario_name)
    train_loader = DataLoader(dataset, shuffle=True)

    input_dim = len(get_tv_rnn_model_features(scenario_name))
    num_layers = trial.suggest_int("num_layer", 1, 20)
    hidden_dims = [trial.suggest_int(f"hidden_dim_{i}", 16, 256) for i in range(num_layers)]
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    drop_out_lstm = trial.suggest_float('drop_out_rate', 0.1, 0.5)
    drop_out_cause = trial.suggest_float('drop_out_rate', 0.1, 0.5)
    llh_loss = trial.suggest_float('llh_loss', 0.1, 1.0)
    ranking_loss = 1 - llh_loss
    num_epochs = 1

    model = DynamicDeepHit(input_dim, hidden_dims, num_risks, drop_out_lstm, drop_out_cause).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        total_loss = 0
        for i, (features, mask, time_to_event, event_indicator, time_to_events, event_indicators) in enumerate(train_loader):
            print_shape = False
            if i == len(train_loader) - 1:
                print_shape = True
            features, mask, time_to_event, event_indicator = [x.to(device) for x in (features, mask, time_to_event, event_indicator)]
            optimizer.zero_grad()

            if print_shape:
                print(f"features shape: {features.shape}")
                print(f"mask shape: {mask.shape}")
                print(f"time_to_event shape: {time_to_event.shape}")
                print(f"event_indicator shape: {event_indicator.shape}")
                print(f"time_to_events shape: {time_to_events.shape}")
                print(f"event_indicators shape: {event_indicators.shape}")

            hazard_preds, _ = model(features, mask, print_shape)
            loss = combine_loss(hazard_preds, time_to_event, event_indicator, num_risks, llh_loss, ranking_loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    c_index = eval_ddh(model, train_loader, device)

    trial.set_user_attr(key="model", value=model)
    return c_index


def eval_ddh(model, data_loader, device):
    c_idxs = []

    for features, mask, time_to_event, event_indicator, time_to_events, event_indicators in data_loader:
        features, mask = features.to(device), mask.to(device)
        hazard_preds, _ = model(features, mask)

        print(f"hazard_preds shape: {hazard_preds.shape}")
        print(f"event_indicator shape: {event_indicator.shape}")
        print(f"time_to_event shape: {time_to_event.shape}")
        print(f"time to events shape {time_to_events.shape}")
        print(f"event_indicators shape {event_indicators.shape}")

        c_idxs.append(calculate_c_index(hazard_preds, time_to_event, event_indicator, num_risks))
        
    avg_c_idx = np.mean(c_idxs, axis=0)
    print(f"Test C-index: {avg_c_idx[0]:.2f}")
    
    return avg_c_idx[0] # 1 risk, which is esrd

def get_device():
    return torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Update the run function to use the device
def run(scenario_name: ExperimentScenario):
    device = get_device()
    _, df_test = get_train_test_data(ExperimentScenario.TIME_VARIANT)

    model_saved_path_dict = {
        ExperimentScenario.TIME_VARIANT: egfr_tv_dynamic_deep_hit_model_path,
        ExperimentScenario.HETEROGENEOUS: hg_dynamic_deep_hit_model_path,
        ExperimentScenario.EGFR_COMPONENTS: egfr_components_dynamic_deep_hit_model_path,
    }
    model_saved_path = model_saved_path_dict[scenario_name]

    if os.path.exists(model_saved_path):
        print("Loading from saved weights")
        model = torch.load(model_saved_path, map_location=device, weights_only = False)
    else:
        model = ex_optuna(lambda trial: objective(trial, scenario_name))
        torch.save(model, model_saved_path)

    model.to(device)

    test_dataset = RNNAttentionDataset(df_test, scenario_name)
    test_dataloader = DataLoader(test_dataset)

    c_idxs = []
    aucs = []

    for features, mask, time_to_event, event_indicators, time_to_events, event_indicators in test_dataloader:
        features, mask = features.to(device), mask.to(device)
        hazard_preds, _ = model(features, mask)
        c_idxs.append(calculate_c_index(hazard_preds, time_to_event, event_indicators, num_risks))

        # calculate mean time-dependent AUC
        times = np.arange(1, 365, 1)

        y_train = Surv.from_arrays(event=event_indicators, time=time_to_events, name_event='has_esrd', name_time='duration_in_days')
        y_test = Surv.from_arrays(event=event_indicators, time=time_to_event, name_event='has_esrd', name_time='duration_in_days')
        _, mean_auc = cumulative_dynamic_auc(y_train, y_test, hazard_preds.cpu().numpy(), times)
        aucs.append(mean_auc)
    
    avg_c_idx = np.mean(c_idxs, axis=0)
    print(f"Test C-index: {avg_c_idx[0]:.2f}")

    avg_auc = np.mean(aucs, axis=0)
    print(f"Mean time-dependent AUC: {avg_auc:.4f}")

    
if __name__ == '__main__':
    run(ExperimentScenario.TIME_VARIANT)
    run(ExperimentScenario.HETEROGENEOUS)
    run(ExperimentScenario.EGFR_COMPONENTS)