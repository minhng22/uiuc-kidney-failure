import math
import pandas as pd
from pkgs.commons import egfr_tv_hazard_transformer_model_path,  hg_hazard_transformer_model_path, egfr_components_hazard_transformer_model_path
from pkgs.data.model_data_store import get_train_test_data
from pkgs.models.hazard_transformer import HazardTransformer
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from pkgs.experiments.utils import ex_optuna, get_tv_rnn_model_features, combine_loss
from pkgs.data.types import ExperimentScenario
from torch.nn.utils.rnn import pad_sequence
from sksurv.metrics import concordance_index_ipcw
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

num_risks = 1

def get_device():
    return torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

class HazardTransformerDataset(Dataset):
    def __init__(self, df, scenario_name: ExperimentScenario):
        self.df = df
        self.subject_groups = list(df.groupby('subject_id'))

        self.scenario_name = scenario_name
        self.features = get_tv_rnn_model_features(scenario_name)

        self.max_seq_length = max(df.groupby('subject_id').size())

    def __len__(self):
        return len(self.subject_groups)

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
                
        return (torch.FloatTensor(features),
                torch.FloatTensor(mask),
                torch.LongTensor([time_to_event]),
                torch.FloatTensor(event),
                torch.FloatTensor(subject_data['duration_in_days'].values),
                torch.FloatTensor(subject_data['has_esrd'].values),
                torch.LongTensor([len(subject_data['duration_in_days'].values)]))
    
def custom_collate_fn(batch):
    features, masks, time_to_events, events, durations, esrds, _ = zip(*batch)

    features = pad_sequence(features, batch_first=True)
    masks = pad_sequence(masks, batch_first=True)
    durations = pad_sequence(durations, batch_first=True)
    esrds = pad_sequence(esrds, batch_first=True)

    return features, masks, torch.stack(time_to_events), torch.stack(events), durations, esrds

def hazard_loss(hazard_preds, delta, time_mask, eps=1e-7):
    p = hazard_preds.clamp(min=eps, max=1-eps)           
    ll1 = delta * torch.log(p)                           
    ll0 = (1 - delta) * torch.log(1 - p)                 
    neg_ll = - (ll1 + ll0) * time_mask.unsqueeze(1)      
    return neg_ll.sum() / (time_mask.sum() * hazard_preds.size(1) + eps)

def objective(trial, scenario_name: ExperimentScenario):
    device = get_device()

    print(f"Running trial {trial.number} for {scenario_name} on device {device}")
    df, _ = get_train_test_data(scenario_name)

    dataset = HazardTransformerDataset(df, scenario_name)
    train_loader = DataLoader(dataset, shuffle=True, collate_fn=custom_collate_fn)

    input_dim = len(get_tv_rnn_model_features(scenario_name))
    num_layers = trial.suggest_int("num_layers", 2, 64)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    drop_out = trial.suggest_float('drop_out_rate', 0.1, 0.5)
    num_epochs = 50
    nhead = trial.suggest_int("n_head", 1, 8)
    nhead_factor = trial.suggest_int("nhead_factor", 1, 16)
    hidden_dims = nhead * nhead_factor

    model = HazardTransformer(input_dim, hidden_dims, num_risks, num_layers, nhead, drop_out).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping parameters
    patience = 5
    best_loss = float('inf')
    patience_counter = 0

    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        total_loss = 0
        for i, (features, mask, time_intervals, event_indicators, _, _) in enumerate(train_loader):
            features, mask, time_intervals, event_indicators = [x.to(device) for x in (features, mask, time_intervals, event_indicators)]
            optimizer.zero_grad()

            hazard_preds, _, _ = model(features, mask)

            batch, _, T = hazard_preds.shape
            t_i = time_intervals.squeeze(1).long()
            arange = torch.arange(T, device=device)
            time_mask = (arange.unsqueeze(0) < t_i.unsqueeze(1)).float()

            delta = torch.zeros_like(hazard_preds)
            for i in range(batch):
                if event_indicators[i].item() == 1:
                    m = t_i[i].item()
                    if m < T:
                        delta[i, 0, m] = 1.0

            loss = hazard_loss(hazard_preds, delta, time_mask)
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

    c_index = c_idx(model, DataLoader(dataset, shuffle=True, collate_fn=custom_collate_fn, batch_size=256), df, device)
    trial.set_user_attr(key="model", value=model)
    return c_index

def c_idx(model, data_loader, train_df, device):
    model.eval()
    y_train = Surv.from_arrays(
        event=train_df['has_esrd'].values.astype(bool),
        time=train_df['duration_in_days'].values,
        name_event='has_esrd',
        name_time='duration_in_days')

    all_scores, all_times, all_events = [], [], []

    with torch.no_grad():
        for X, mask, times, events, _, _ in data_loader:
            X, mask = X.to(device), mask.to(device)
            hazard_preds, _, _ = model(X, mask)              
            # Compute cumulative incidence for event 0 by time horizon T
            surv = torch.cumprod(1 - hazard_preds.sum(dim=2), dim=1)         
            cumidx = 1 - surv
            risk = cumidx[:, -1].cpu().numpy()               

            all_scores.extend(risk.tolist())
            all_times.extend(times.squeeze(1).cpu().numpy().tolist())
            all_events.extend(events.squeeze(1).cpu().numpy().tolist())

    c_td, _, _, _, _ = concordance_index_ipcw(
        y_train,
        Surv.from_arrays(event=np.array(all_events).astype(bool), time=np.array(all_times), name_event='has_esrd', name_time='duration_in_days'),
        all_scores)
    print(f"Time-dependent C-index: {c_td:.4f}")
    return c_td

def auc(model: HazardTransformer, train_df, dataloader: DataLoader, device):
    y_train = Surv.from_arrays(
        event=train_df['has_esrd'].values, time=train_df['duration_in_days'].values, name_event='has_esrd', name_time='duration_in_days')
    aucs = []
    times = np.arange(1, 365, 1)
    for features, mask, time_to_events, event_indicators, _, _ in dataloader:
        features, mask = features.to(device), mask.to(device)
        y_test = Surv.from_arrays(
            event=event_indicators.squeeze(),
            time=time_to_events.squeeze(),
            name_event='has_esrd',
            name_time='duration_in_days'
        )

        hazard_preds, _, _ = model(features, mask)
        hazard_preds = hazard_preds[:, 0, 0].detach().cpu().numpy()

        _, mean_auc = cumulative_dynamic_auc(y_train, y_test, hazard_preds, times)
        aucs.append(mean_auc)

    avg_auc = np.mean(aucs, axis=0)
    print(f"Mean time-dependent AUC: {avg_auc:.2f}")
    

def run(scenario_name: ExperimentScenario):
    device = get_device()
    df, df_test = get_train_test_data(scenario_name)

    model_saved_path_dict = {
        ExperimentScenario.TIME_VARIANT: egfr_tv_hazard_transformer_model_path,
        ExperimentScenario.HETEROGENEOUS: hg_hazard_transformer_model_path,
        ExperimentScenario.EGFR_COMPONENTS: egfr_components_hazard_transformer_model_path,
    }
    model_saved_path = model_saved_path_dict[scenario_name]
    
    if os.path.exists(model_saved_path):
        print("Loading from saved weights")
        model = torch.load(model_saved_path, map_location=device, weights_only=False)
    else:
        model = ex_optuna(lambda trial: objective(trial, scenario_name))
        torch.save(model, model_saved_path)
    
    model.to(device)

    print("model summary")
    print(model)

    c_idx(model, DataLoader(HazardTransformerDataset(df_test, scenario_name), shuffle=True, collate_fn=custom_collate_fn, batch_size=256), df, device)
    auc(model, df, DataLoader(HazardTransformerDataset(df_test, scenario_name), shuffle=True, collate_fn=custom_collate_fn, batch_size=256), device)

if __name__ == '__main__':
    run(ExperimentScenario.TIME_VARIANT)
    run(ExperimentScenario.HETEROGENEOUS)
    run(ExperimentScenario.EGFR_COMPONENTS)