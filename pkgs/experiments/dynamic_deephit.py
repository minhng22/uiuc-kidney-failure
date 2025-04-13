import pandas as pd
from pkgs.commons import egfr_tv_dynamic_deep_hit_model_path, hg_dynamic_deep_hit_model_path, egfr_components_dynamic_deep_hit_model_path
from pkgs.data.model_data_store import get_train_test_data
from pkgs.models.dynamicdeephit import DynamicDeepHit
import torch
from torch.utils.data import DataLoader, Dataset

from pkgs.experiments.utils import ex_optuna, get_tv_rnn_model_features, combine_loss
from pkgs.data.types import ExperimentScenario

import os
import numpy as np
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc
from lifelines.utils import concordance_index
from torchsummary import summary


num_risks = 1 # esrd

class DynamicDeepHitDataset(Dataset):
    def __init__(self, df, scenario_name: ExperimentScenario):
        self.df = df
        self.subject_groups = list(df.groupby('subject_id'))

        self.scenario_name = scenario_name
        self.features = get_tv_rnn_model_features(scenario_name)

        self.max_seq_length = max(df.groupby('subject_id').size())

    def __len__(self):
        return len(self.subject_groups)

    def get_all_subj_data(self):
        feats, masks, tte, ev, ttes, inds = torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

        for i in range(len(self.subject_groups)):
            f_i, m_i, tte_i, ev_i, ttes_i, ind_i = self.__getitem__(i)
            feats = torch.concat((feats, f_i), dim=0)
            masks = torch.concat((masks, m_i), dim=0)
            tte = torch.concat((tte, tte_i), dim=0)
            ev = torch.concat((ev, ev_i), dim=0)
            ttes = torch.concat((ttes, ttes_i), dim=0)
            inds = torch.concat((inds, ind_i), dim=0)

        print(f"feats shape: {feats.shape}")
        print(f"masks shape: {masks.shape}")
        print(f"tte shape: {tte.shape}")
        print(f"ev shape: {ev.shape}")
        print(f"ttes shape: {ttes.shape}")
        print(f"inds shape: {inds.shape}")

        return (
            feats, masks, tte, ev, ttes, inds
        )

    def __getitem__(self, idx):
        _, subject_data = self.subject_groups[idx]
        seq_length = len(subject_data)

        assert isinstance(subject_data, pd.DataFrame), f"subject_data is not a DataFrame: {type(subject_data)}"
        assert subject_data['duration_in_days'].is_monotonic_increasing, "subject_data is not sorted by time"
        
        features = np.zeros((self.max_seq_length, len(self.features)))
        mask = np.zeros(self.max_seq_length)
        
        if self.scenario_name == ExperimentScenario.TIME_VARIANT:
            features[:seq_length, 0] = (subject_data['egfr'].values - self.df['egfr'].mean()) / self.df['egfr'].std()
        elif self.scenario_name == ExperimentScenario.HETEROGENEOUS:
            features[:seq_length, 0] = (subject_data['egfr'].values - self.df['egfr'].mean()) / self.df['egfr'].std()
            features[:seq_length, 1] = subject_data['egfr_missing'].values
            features[:seq_length, 2] = (subject_data['protein'].values - self.df['protein'].mean()) / self.df['protein'].std()
            features[:seq_length, 3] = subject_data['protein_missing'].values
            features[:seq_length, 4] = (subject_data['albumin'].values - self.df['albumin'].mean()) / self.df['albumin'].std()
            features[:seq_length, 5] = subject_data['albumin_missing'].values
        elif self.scenario_name == ExperimentScenario.EGFR_COMPONENTS:
            features[:seq_length, 0] = (subject_data['age'].values - self.df['age'].mean()) / self.df['age'].std()
            features[:seq_length, 1] = subject_data['gender'].values
            features[:seq_length, 2] = (subject_data['serum_creatinine'].values - self.df['serum_creatinine'].mean()) / self.df['serum_creatinine'].std()
        
        mask[:seq_length] = 1
        
        time_to_event = subject_data['duration_in_days'].iloc[-1]
        event = np.array([subject_data['has_esrd'].iloc[-1]])
        time_to_events = subject_data['duration_in_days'].values
        event_indicators = subject_data['has_esrd'].values
                
        return (torch.FloatTensor(features),
                torch.FloatTensor(mask),
                torch.LongTensor([time_to_event]),
                torch.FloatTensor(event),
                torch.FloatTensor(time_to_events),
                torch.FloatTensor(event_indicators))

def objective(trial, scenario_name: ExperimentScenario):
    device = get_device()

    print(f"Running trial {trial.number} for {scenario_name} on device {device}")
    df, _ = get_train_test_data(scenario_name)

    dataset = DynamicDeepHitDataset(df, scenario_name)
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

    if os.path.exists(egfr_tv_dynamic_deep_hit_model_path):
        print("Loading from saved weights")
        model = torch.load(egfr_tv_dynamic_deep_hit_model_path, map_location=device, weights_only = False)
    else:
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
        torch.save(model, egfr_tv_dynamic_deep_hit_model_path)

    c_index = c_idx(model, dataset, device)

    trial.set_user_attr(key="model", value=model)
    return c_index


def c_idx(model: DynamicDeepHit, dataset: DynamicDeepHitDataset, device):
    features, mask, _, _, time_to_events, event_indicators = dataset.get_all_subj_data()

    features, mask = features.to(device), mask.to(device)

    features = features.contiguous()
    mask = mask.contiguous()

    summary(model)

    hazard_preds, _ = model(features, mask, True)

    print(f"hazard_preds shape: {hazard_preds.shape}")
    print(f"time to events shape {time_to_events.shape}")
    print(f"event_indicators shape {event_indicators.shape}")

    hazard_preds = hazard_preds.cpu().detach().numpy()
    hazard_preds = hazard_preds[:, 0, :] # only one risk, which is esrd
    final_risk_scores = []
    for i in range(time_to_events.shape[1]):
        duration = time_to_events[:, i]
        risk_at_time = hazard_preds[:, int(duration)]
        final_risk_scores.append(risk_at_time)
        
    final_risk_scores = np.array(final_risk_scores).reshape((1, -1))
    print(f"final_risk_scores shape: {final_risk_scores}")
    print(f"time_to_events: {time_to_events}")
    print(f"event_indicators: {event_indicators}")
    
    c_idx = concordance_index(time_to_events, final_risk_scores, event_indicators)
    print(f"Test C-index: {c_idx:.2f}")
    
    return c_idx

def get_device():
    return torch.device("cpu")

# Update the run function to use the device
def run(scenario_name: ExperimentScenario):
    torch.backends.cudnn.enabled = False
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
    test_dataset = DynamicDeepHitDataset(df_test, scenario_name)

    c_idx(model, test_dataset, device)

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