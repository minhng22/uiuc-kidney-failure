import pandas as pd
from pkgs.commons import egfr_tv_dynamic_deep_hit_model_path, hg_dynamic_deep_hit_model_path, egfr_components_dynamic_deep_hit_model_path, fivelabms_dynamic_deep_hit_model_path, ckd_fifty_features_heterogeneous_dynamic_deep_hit_model_path
from pkgs.data_analysis.model_data_store import get_train_test_data
from pkgs.models.dynamicdeephit import DynamicDeepHit
import torch
from torch.utils.data import DataLoader, Dataset

from pkgs.experiments.utils import ex_optuna, get_tv_rnn_model_features, combine_loss, compute_brier_score_from_risk_scores
from pkgs.data_analysis.types import ExperimentScenario

import os
import numpy as np
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored
from pkgs.experiments.utils import get_device

num_risks = 1 # esrd
model_saved_path_dict = {
        ExperimentScenario.TIME_VARIANT: egfr_tv_dynamic_deep_hit_model_path,
        ExperimentScenario.HETEROGENEOUS: hg_dynamic_deep_hit_model_path,
        ExperimentScenario.EGFR_COMPONENTS: egfr_components_dynamic_deep_hit_model_path,
        ExperimentScenario.FIVELABMS: fivelabms_dynamic_deep_hit_model_path,
        ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS: ckd_fifty_features_heterogeneous_dynamic_deep_hit_model_path,
    }

class DynamicDeepHitDataset(Dataset):
    def __init__(self, df, scenario_name: ExperimentScenario):
        self.df = df
        self.subject_groups = list(df.groupby('subject_id'))

        self.scenario_name = scenario_name
        self.features = get_tv_rnn_model_features(scenario_name)

        self.max_seq_length = max(df.groupby('subject_id').size())
    
    def number_of_subjects(self):
        return len(self.subject_groups)

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
        elif self.scenario_name == ExperimentScenario.FIVELABMS:
            features[:seq_length, 0] = (subject_data['egfr'].values - self.df['egfr'].mean()) / self.df['egfr'].std()
            features[:seq_length, 1] = subject_data['egfr_missing'].values
            features[:seq_length, 2] = (subject_data['hemoglobin'].values - self.df['hemoglobin'].mean()) / self.df['hemoglobin'].std()
            features[:seq_length, 3] = subject_data['hemoglobin_missing'].values
        elif self.scenario_name == ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS:
            # 50 lab features with missingness indicators (100 features total)
            lab_names = ['egfr', 'urea_nitrogen', 'hemoglobin', 'serum_albumin', 'potassium', 
                         'sodium', 'bicarbonate', 'phosphate', 'calcium', 'glucose',
                         'chloride', 'anion_gap', 'hematocrit', 'platelet_count', 'wbc',
                         'rbc', 'mcv', 'mch', 'mchc', 'rdw', 'magnesium', 'uric_acid',
                         'bilirubin_total', 'alt', 'ast', 'alkaline_phosphatase', 'ldh',
                         'iron', 'total_protein', 'cholesterol_total', 'triglycerides',
                         'inr', 'ptt', 'crp', 'ferritin', 'transferrin', 'tibc', 'lactate',
                         'base_excess', 'pco2', 'po2', 'ph', 'bilirubin_direct', 'bilirubin_indirect',
                         'ggt', 'amylase', 'lipase', 'ck', 'troponin', 'bnp']
            feat_idx = 0
            for lab_name in lab_names:
                features[:seq_length, feat_idx] = (subject_data[lab_name].values - self.df[lab_name].mean()) / (self.df[lab_name].std() + 1e-8)
                features[:seq_length, feat_idx + 1] = subject_data[f'{lab_name}_missing'].values
                feat_idx += 2
        
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
    train_loader = DataLoader(dataset, shuffle=True, batch_size=256)

    input_dim = len(get_tv_rnn_model_features(scenario_name))
    num_layers = trial.suggest_int("num_layer", 1, 4)
    hidden_dims = [trial.suggest_int(f"hidden_dim_{i}", 16, 128) for i in range(num_layers)]
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    drop_out_lstm = trial.suggest_float('drop_out_rate', 0.1, 0.5)
    drop_out_cause = trial.suggest_float('drop_out_rate', 0.1, 0.5)
    llh_loss = trial.suggest_float('llh_loss', 0.1, 1.0)
    ranking_loss = 1 - llh_loss
    num_epochs = 50

    model = DynamicDeepHit(input_dim, hidden_dims, num_risks, drop_out_lstm, drop_out_cause).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping parameters
    patience = 5
    best_loss = float('inf')
    patience_counter = 0

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
    
    c_index = c_idx(model, dataset, device)

    saved_path = model_saved_path_dict[scenario_name]
    if os.path.exists(saved_path):
        saved_model = torch.load(saved_path, map_location=device)
        saved_cidx = c_idx(saved_model, dataset, device)
        
        print(f"Saved model C-index: {saved_cidx:.4f}, Current trial C-index: {c_index:.4f}")
        if c_index > saved_cidx:
            print("New model is better, saving current model")
            torch.save(model, saved_path)
        else:
            print("Saved model is better, keeping it")
    else:
        print("No existing model, saving current model")
        torch.save(model, saved_path)

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
        # if i == 0:
        #     debug_mode = True
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
                f_time_to_events = time_to_events[j][:p_seq_len].cpu().detach().numpy()
                f_risk_scores = hazard_preds[j][:p_seq_len]
                f_event_indicators = event_indicators[j][:p_seq_len].cpu().detach().numpy()
            else:
                f_time_to_events = np.concatenate((f_time_to_events, time_to_events[j][:p_seq_len].cpu().detach().numpy()), axis=0)
                f_risk_scores = np.concatenate((f_risk_scores, hazard_preds[j][:p_seq_len]), axis=0)
                f_event_indicators = np.concatenate((f_event_indicators, event_indicators[j][:p_seq_len].cpu().detach().numpy()), axis=0)
        
        if i == 0:
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

def brier_score_evaluation(model: DynamicDeepHit, test_dataset: DynamicDeepHitDataset, train_df: pd.DataFrame, device):
    """Compute Brier Score for Dynamic DeepHit model"""
    dataloader = DataLoader(test_dataset, shuffle=False, batch_size=256)
    
    all_risk_scores = []
    all_times = []
    all_events = []
    
    for features, mask, time_to_event, event_indicator, time_to_events, event_indicators, seq_lens in dataloader:
        features, mask, time_to_event, event_indicator, time_to_events, event_indicators, seq_lens = [x.to(device) for x in (features, mask, time_to_event, event_indicator, time_to_events, event_indicators, seq_lens)]
        
        hazard_preds, _ = model(features, mask)
        hazard_preds = hazard_preds.cpu().detach().numpy()
        hazard_preds = hazard_preds[:, 0, :]  # only one risk (ESRD)
        
        for j in range(hazard_preds.shape[0]):
            p_seq_len = int(seq_lens[j])
            all_risk_scores.extend(hazard_preds[j][:p_seq_len])
            all_times.extend(time_to_events[j][:p_seq_len].cpu().detach().numpy())
            all_events.extend(event_indicators[j][:p_seq_len].cpu().detach().numpy())
    
    # Create test dataframe from collected data
    test_df = pd.DataFrame({
        'duration_in_days': all_times,
        'has_esrd': all_events
    })
    
    # Compute Brier Score
    brier_score = compute_brier_score_from_risk_scores(train_df, test_df, np.array(all_risk_scores))
    if brier_score is not None:
        print(f'Integrated Brier Score Test: {brier_score}')
    
    return brier_score

def simple_cindex(times, predictions, events):
    pairs = 0
    concordant = 0
    tied = 0
    
    print(f"Starting simple C-index calculation...")
    
    for i in range(len(times)):
        if not events[i]:
            continue
            
        for j in range(len(times)):
            if i == j:
                continue
                
            if times[j] > times[i]:
                pairs += 1
                if predictions[j] <= predictions[i]:
                    concordant += 1
    
    print(f"Pairs evaluated: {pairs}, Concordant: {concordant}")
    
    if pairs == 0:
        print("WARNING: No valid pairs found for comparison!")
        return 0.0
        
    return (concordant + tied) / pairs

def c_idx(model: DynamicDeepHit, dataset: DynamicDeepHitDataset, device, test=False):
    model.eval()

    loader = DataLoader(dataset, shuffle=False, batch_size=256)
    
    all_times = []
    all_events = []
    all_risks = []

    with torch.no_grad():
        debug_print = True
        for features, mask, T, E, _, _, _ in loader:
            features, mask = features.to(device), mask.to(device)
            hazards, _ = model(features, mask, False)
            hazards = hazards[:, 0, :].cpu().numpy()

            if debug_print:
                print(f"features shape: {features.shape}, mask shape: {mask.shape}, T shape: {T.shape}, E shape: {E.shape} hazards shape: {hazards.shape}")
                
            T = T.cpu().numpy().ravel().astype(int)
            E = E.cpu().numpy().ravel().astype(bool)

            for j, t_j in enumerate(T):
                if debug_print:
                    print(f"Processing subject {j} with time {t_j} and event {E[j]} hazards shape: {hazards[j, t_j : t_j + 1]}")
                    debug_print = False
                all_times.append(t_j)
                all_events.append(E[j])
                all_risks.append(hazards[j, t_j : t_j + 1])

    all_times = np.array(all_times)
    all_events = np.array(all_events)
    all_risks = np.array(all_risks).flatten()
    all_surv = 1.0 - all_risks

    print(f"all_E shape: {all_events.shape} first 10 values: {all_events[:10]}")
    print(f"all_R shape: {all_risks.shape} first 10 values: {all_risks[:10]}")
    print(f"all_T shape: {all_times.shape} first 10 values: {all_times[:10]}")

    cindex = concordance_index(all_times, all_surv, all_events)

    res = concordance_index_censored(all_events, all_times, all_risks)
    print(f"Global test C-index (scikit-survival): {res[0]:.3f} number of concordant pairs: {res[1]}")
    
    print(f"Global test C-index: {cindex:.3f}")

    if test:
        num_events = sum(all_events)
        print(f"Event rate: {num_events}/{len(all_events)} ({num_events/len(all_events)*100:.2f}%)")
        
        simple_ci = simple_cindex(all_times, all_risks, all_events)
        print(f"Simple C-index implementation: {simple_ci:.3f}")
    return cindex

# Update the run function to use the device
def run(scenario_name: ExperimentScenario):
    torch.backends.cudnn.enabled = False
    device = get_device()
    df, df_test = get_train_test_data(scenario_name)

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

    c_idx(model, test_dataset, device, True)
    auc(model, test_dataset, df, device)
    brier_score_evaluation(model, test_dataset, df, device)

if __name__ == '__main__':
    #run(ExperimentScenario.TIME_VARIANT)
    #run(ExperimentScenario.HETEROGENEOUS)
    #run(ExperimentScenario.EGFR_COMPONENTS)
    #run(ExperimentScenario.FIVELABMS)
    run(ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS)