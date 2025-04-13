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
        feats, masks, tte, ev, ttes, inds = [None for _ in range(6)]

        for i in range(len(self.subject_groups)):
            f_i, m_i, tte_i, ev_i, ttes_i, ind_i = self.__getitem__(i)
            if feats is None:
                feats = f_i.unsqueeze(0)
                masks = m_i.unsqueeze(0)
                tte = tte_i.unsqueeze(0)
                ev = ev_i.unsqueeze(0)
                ttes = ttes_i.unsqueeze(0)
                inds = ind_i.unsqueeze(0)
                print(f"feats shape: {feats.shape}")
                print(f"masks shape: {masks.shape}")
                print(f"tte shape: {tte.shape}")
            else:
                feats = torch.concat((feats, f_i.unsqueeze(0)), dim=0)
                masks = torch.concat((masks, m_i.unsqueeze(0)), dim=0)
                tte = torch.concat((tte, tte_i.unsqueeze(0)), dim=0)
                ev = torch.concat((ev, ev_i.unsqueeze(0)), dim=0)
                ttes = torch.concat((ttes, ttes_i.unsqueeze(0)), dim=0)
                inds = torch.concat((inds, ind_i.unsqueeze(0)), dim=0)

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

        time_to_events = np.zeros((self.max_seq_length))
        time_to_events[:len(subject_data['duration_in_days'].values)] = subject_data['duration_in_days'].values

        event_indicators = np.zeros((self.max_seq_length))
        event_indicators[:len(subject_data['has_esrd'].values)] = subject_data['has_esrd'].values
                
        return (torch.FloatTensor(features),
                torch.FloatTensor(mask),
                torch.LongTensor([time_to_event]),
                torch.FloatTensor(event),
                torch.FloatTensor(time_to_events),
                torch.FloatTensor(event_indicators),
                torch.LongTensor([len(subject_data['has_esrd'].values)]))

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
    num_epochs = 50

    model = DynamicDeepHit(input_dim, hidden_dims, num_risks, drop_out_lstm, drop_out_cause).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        total_loss = 0
        for i, (features, mask, time_to_event, event_indicator, time_to_events, event_indicators, seq_lens) in enumerate(train_loader):
            debug_mode = False
            if i == 0:
                debug_mode = True
            features, mask, time_to_event, event_indicator = [x.to(device) for x in (features, mask, time_to_event, event_indicator)]
            optimizer.zero_grad()

            if debug_mode:
                print(f"features shape: {features.shape}, mask shape: {mask.shape}, time_to_event shape: {time_to_event.shape}, "
                      f"event_indicator shape: {event_indicator.shape}, time_to_events shape: {time_to_events.shape}, "
                      f"event_indicators shape: {event_indicators.shape}, sequence lengths shape: {seq_lens.shape}")

            hazard_preds, _ = model(features, mask, debug_mode)
            loss = combine_loss(hazard_preds, time_to_event, event_indicator, num_risks, llh_loss, ranking_loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    c_index = c_idx(model, dataset, device)

    trial.set_user_attr(key="model", value=model)
    return c_index

def auc(model: DynamicDeepHit, test_dataset: DynamicDeepHitDataset, train_df: pd.DataFrame, device):
    times = np.arange(1, 365, 1)
    y_train = Surv.from_arrays(
        event=train_df['has_esrd'].values, time=train_df['duration_in_days'].values, name_event='has_esrd', name_time='duration_in_days')

    dataloader = DataLoader(test_dataset, shuffle=False, batch_size=256)
    aucs = []

    for i, (features, mask, time_to_event, event_indicator, time_to_events, event_indicators, seq_lens) in enumerate(dataloader):
        debug_mode = False
        if i == 0:
            debug_mode = True
        features, mask, time_to_event, event_indicator, time_to_events, event_indicators, seq_lens = [x.to(device) for x in (features, mask, time_to_event, event_indicator, time_to_events, event_indicators, seq_lens)]
        
        if debug_mode:
            print(f"features shape: {features.shape}")
            print(f"mask shape: {mask.shape}")
            print(f"time_to_event shape: {time_to_event.shape}")
            print(f"time_to_events shape: {time_to_events.shape}")
            print(f"seq_lens shape: {seq_lens.shape}")

        hazard_preds, _ = model(features, mask, debug_mode)

        hazard_preds = hazard_preds.cpu().detach().numpy()
        hazard_preds = hazard_preds[:, 0, :] # only one risk, which is esrd

        if debug_mode:
            print(f"calc hazard_preds shape: {hazard_preds.shape}")
        
        f_time_to_events, f_risk_scores, f_event_indicators = None, None, None

        for j in range(hazard_preds.shape[0]):
            p_seq_len = int(seq_lens[j])
            if f_time_to_events is None:
                f_time_to_events = time_to_events[j][:p_seq_len]
                f_risk_scores = hazard_preds[j][:p_seq_len]
                f_event_indicators = event_indicators[j][:p_seq_len]
            else:
                f_time_to_events = np.concatenate((f_time_to_events, time_to_events[j][:p_seq_len]), axis=0)
                f_risk_scores = np.concatenate((f_risk_scores, hazard_preds[j][:p_seq_len]), axis=0)
                f_event_indicators = np.concatenate((f_event_indicators, event_indicators[j][:p_seq_len]), axis=0)
        
        if debug_mode:
            print(f"f_time_to_events shape: {len(f_time_to_events)}")
            print(f"f_risk_scores shape: {len(f_risk_scores)}")
            print(f"f_event_indicators shape: {len(f_event_indicators)}")

        y_test = Surv.from_arrays(event=f_event_indicators, time=f_time_to_events, name_event='has_esrd', name_time='duration_in_days')
        _, mean_auc = cumulative_dynamic_auc(y_train, y_test, f_risk_scores, times)
        aucs.append(mean_auc)

        if debug_mode:
            print(f"Mean AUC: {mean_auc}")

    avg_auc = np.mean(aucs, axis=0)
    print(f"Mean time-dependent AUC: {avg_auc:.2f}")

def c_idx(model: DynamicDeepHit, dataset: DynamicDeepHitDataset, device):
    dataloader = DataLoader(dataset, shuffle=False, batch_size=256)
    c_idxs = []
    for i, (features, mask, time_to_event, event_indicator, time_to_events, event_indicators, seq_lens) in enumerate(dataloader):
        debug_mode = False
        if i == 0:
            debug_mode = True
        features, mask, time_to_event, event_indicator, time_to_events, event_indicators, seq_lens = [x.to(device) for x in (features, mask, time_to_event, event_indicator, time_to_events, event_indicators, seq_lens)]
        if debug_mode:
            print(f"features shape: {features.shape}")
            print(f"mask shape: {mask.shape}")
            print(f"time_to_event shape: {time_to_event.shape}")
            print(f"time_to_events shape: {time_to_events.shape}")
            print(f"seq_lens shape: {seq_lens.shape}")

        hazard_preds, _ = model(features, mask, debug_mode)

        hazard_preds = hazard_preds.cpu().detach().numpy()
        hazard_preds = hazard_preds[:, 0, :] # only one risk, which is esrd

        if debug_mode:
            print(f"calc hazard_preds shape: {hazard_preds.shape}")
        
        f_time_to_events, f_risk_scores, f_event_indicators = None, None, None

        for j in range(hazard_preds.shape[0]):
            p_seq_len = int(seq_lens[j])
            if f_time_to_events is None:
                f_time_to_events = time_to_events[j][:p_seq_len]
                f_risk_scores = hazard_preds[j][:p_seq_len]
                f_event_indicators = event_indicators[j][:p_seq_len]
            else:
                f_time_to_events = np.concatenate((f_time_to_events, time_to_events[j][:p_seq_len]), axis=0)
                f_risk_scores = np.concatenate((f_risk_scores, hazard_preds[j][:p_seq_len]), axis=0)
                f_event_indicators = np.concatenate((f_event_indicators, event_indicators[j][:p_seq_len]), axis=0)
        
        if debug_mode:
            print(f"f_time_to_events shape: {len(f_time_to_events)}")
            print(f"f_risk_scores shape: {len(f_risk_scores)}")
            print(f"f_event_indicators shape: {len(f_event_indicators)}")

        c_idx = concordance_index(f_time_to_events, f_risk_scores, f_event_indicators)            
        c_idxs.append(c_idx)

        if debug_mode:
            print(f"Concordance index: {c_idx:.2f}")

    c_idx = np.mean(c_idxs, axis=0)
    print(f"Test C-index: {c_idx:.2f}")
    return np.mean(c_idxs, axis=0)

def get_device():
    return torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Update the run function to use the device
def run(scenario_name: ExperimentScenario):
    torch.backends.cudnn.enabled = False
    device = get_device()
    df, df_test = get_train_test_data(ExperimentScenario.TIME_VARIANT)

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
    print("model summary:")
    print(model)
    test_dataset = DynamicDeepHitDataset(df_test, scenario_name)

    c_idx(model, test_dataset, device)
    auc(model, test_dataset, df, device)

    
if __name__ == '__main__':
    run(ExperimentScenario.TIME_VARIANT)
    run(ExperimentScenario.HETEROGENEOUS)
    run(ExperimentScenario.EGFR_COMPONENTS)