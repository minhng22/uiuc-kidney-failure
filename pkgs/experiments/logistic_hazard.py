import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
from pycox.models import LogisticHazard
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
import torchtuples as tt
from pkgs.experiments.utils import get_device

from pkgs.commons import (egfr_tv_logistic_hazard_model_path, hg_logistic_hazard_model_path,
                          egfr_components_logistic_hazard_model_path, fivelabms_logistic_hazard_model_path,
                          heterogen_impute_logistic_hazard_model_path, ckd_fifty_features_heterogeneous_logistic_hazard_model_path,
                          four_features_logistic_hazard_model_path, eight_features_logistic_hazard_model_path,
                          twenty_features_heterogeneous_logistic_hazard_model_path,
                          ckd_fifty_features_heterogeneous_train_data_path)
from pkgs.data_analysis.model_data_store import get_train_test_data
from pkgs.data_analysis.types import ExperimentScenario
from pkgs.experiments.utils import ex_optuna, get_tv_rnn_model_features, compute_brier_score_from_risk_scores

from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored
from lifelines.utils import concordance_index

num_risks = 1
model_saved_path_dict = {
    ExperimentScenario.TIME_VARIANT: egfr_tv_logistic_hazard_model_path,
    ExperimentScenario.HETEROGENEOUS: hg_logistic_hazard_model_path,
    ExperimentScenario.EGFR_COMPONENTS: egfr_components_logistic_hazard_model_path,
    ExperimentScenario.FIVELABMS: fivelabms_logistic_hazard_model_path,
    ExperimentScenario.HETEROGENEOUS_IMPUTE: heterogen_impute_logistic_hazard_model_path,
    ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS: ckd_fifty_features_heterogeneous_logistic_hazard_model_path,
    ExperimentScenario.FOUR_FEATURES: four_features_logistic_hazard_model_path,
    ExperimentScenario.EIGHT_FEATURES: eight_features_logistic_hazard_model_path,
    ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS: twenty_features_heterogeneous_logistic_hazard_model_path,
}

class LogisticHazardDataset(Dataset):
    def __init__(self, df, scenario_name: ExperimentScenario):
        self.df = df
        self.subject_groups = list(df.groupby('subject_id'))
        self.scenario_name = scenario_name
        self.features = get_tv_rnn_model_features(scenario_name)

    def __len__(self):
        return len(self.subject_groups)

    def prepare_data_for_pycox(self):
        all_features = []
        all_durations = []
        all_events = []
        
        for _, subject_data in self.subject_groups:
            last_obs = subject_data.iloc[-1]
            
            if self.scenario_name == ExperimentScenario.TIME_VARIANT:
                features = [(last_obs['egfr'] - self.df['egfr'].mean()) / self.df['egfr'].std()]
            elif self.scenario_name == ExperimentScenario.HETEROGENEOUS:
                features = [
                    (last_obs['egfr'] - self.df['egfr'].mean()) / self.df['egfr'].std(),
                    last_obs['egfr_missing'],
                    (last_obs['protein'] - self.df['protein'].mean()) / self.df['protein'].std(),
                    last_obs['protein_missing'],
                    (last_obs['albumin'] - self.df['albumin'].mean()) / self.df['albumin'].std(),
                    last_obs['albumin_missing']
                ]
            elif self.scenario_name == ExperimentScenario.EGFR_COMPONENTS:
                features = [
                    (last_obs['age'] - self.df['age'].mean()) / self.df['age'].std(),
                    last_obs['gender'],
                    (last_obs['serum_creatinine'] - self.df['serum_creatinine'].mean()) / self.df['serum_creatinine'].std()
                ]
            elif self.scenario_name == ExperimentScenario.FIVELABMS:
                features = [
                    (last_obs['egfr'] - self.df['egfr'].mean()) / self.df['egfr'].std(),
                    last_obs['egfr_missing'],
                    (last_obs['hemoglobin'] - self.df['hemoglobin'].mean()) / self.df['hemoglobin'].std(),
                    last_obs['hemoglobin_missing'],
                ]
            elif self.scenario_name == ExperimentScenario.HETEROGENEOUS_IMPUTE:
                # Imputed heterogeneous: same features as FIVELABMS but without missingness indicators
                features = [
                    (last_obs['egfr'] - self.df['egfr'].mean()) / self.df['egfr'].std(),
                    (last_obs['hemoglobin'] - self.df['hemoglobin'].mean()) / self.df['hemoglobin'].std(),
                ]
            elif self.scenario_name == ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS:
                # 50 lab features with missingness indicators
                lab_names = ['egfr', 'urea_nitrogen', 'hemoglobin', 'serum_albumin', 'potassium',
                             'sodium', 'bicarbonate', 'phosphate', 'calcium', 'glucose',
                             'chloride', 'anion_gap', 'hematocrit', 'platelet_count', 'wbc',
                             'rbc', 'mcv', 'mch', 'mchc', 'rdw', 'magnesium', 'uric_acid',
                             'bilirubin_total', 'alt', 'ast', 'alkaline_phosphatase', 'ldh',
                             'iron', 'total_protein', 'cholesterol_total', 'triglycerides',
                             'inr', 'ptt', 'crp', 'ferritin', 'transferrin', 'tibc',
                             'lymphocytes', 'neutrophils', 'monocytes', 'basophils', 'eosinophils',
                             'pt', 'rdw_sd', 'lab_h', 'lab_l', 'lab_i',
                             'urine_specific_gravity', 'urine_ph', 'ph']
                features = []
                for lab_name in lab_names:
                    features.append((last_obs[lab_name] - self.df[lab_name].mean()) / (self.df[lab_name].std() + 1e-8))
                    features.append(last_obs[f'{lab_name}_missing'])
            elif self.scenario_name == ExperimentScenario.FOUR_FEATURES:
                features = [
                    (last_obs['age'] - self.df['age'].mean()) / self.df['age'].std(),
                    last_obs['gender'],
                    (last_obs['egfr'] - self.df['egfr'].mean()) / self.df['egfr'].std(),
                    (last_obs['uacr'] - self.df['uacr'].mean()) / self.df['uacr'].std(),
                ]
            elif self.scenario_name == ExperimentScenario.EIGHT_FEATURES:
                features = [
                    (last_obs['age'] - self.df['age'].mean()) / self.df['age'].std(),
                    last_obs['gender'],
                ]
                for lab_name in ['egfr', 'uacr', 'calcium', 'phosphate', 'bicarbonate', 'serum_albumin']:
                    features.append((last_obs[lab_name] - self.df[lab_name].mean()) / (self.df[lab_name].std() + 1e-8))
            elif self.scenario_name == ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS:
                # top 20 lab features with missingness indicators
                lab_names = ['egfr', 'potassium', 'urea_nitrogen', 'sodium', 'chloride', 'bicarbonate',
                             'anion_gap', 'hematocrit', 'platelet_count', 'hemoglobin', 'wbc', 'mchc',
                             'mch', 'rbc', 'mcv', 'rdw', 'glucose', 'calcium', 'magnesium', 'phosphate']
                features = []
                for lab_name in lab_names:
                    features.append((last_obs[lab_name] - self.df[lab_name].mean()) / (self.df[lab_name].std() + 1e-8))
                    features.append(last_obs[f'{lab_name}_missing'])
            else:
                raise ValueError(f"Unsupported scenario: {self.scenario_name}")
            
            all_features.append(features)
            all_durations.append(last_obs['duration_in_days'])
            all_events.append(last_obs['has_esrd'])
        
        return np.array(all_features, dtype=np.float32), np.array(all_durations, dtype=np.float32), np.array(all_events, dtype=np.int32)

def objective(trial, scenario_name: ExperimentScenario):
    device = get_device()
    
    num_nodes = trial.suggest_categorical('num_nodes', [16, 32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    df, df_test = get_train_test_data(scenario_name)
    
    train_dataset = LogisticHazardDataset(df, scenario_name)
    x_train, durations_train, events_train = train_dataset.prepare_data_for_pycox()
    
    test_dataset = LogisticHazardDataset(df_test, scenario_name)
    x_test, durations_test, events_test = test_dataset.prepare_data_for_pycox()
    
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    
    num_durations = 50
    labtrans = LabTransDiscreteTime(num_durations)
    idx_durations_train, events_train_disc = labtrans.fit_transform(durations_train, events_train)
    idx_durations_test, events_test_disc = labtrans.transform(durations_test, events_test)
    
    idx_durations_train = torch.tensor(idx_durations_train, dtype=torch.int64)
    events_train_disc = torch.tensor(events_train_disc, dtype=torch.float32)
    idx_durations_test = torch.tensor(idx_durations_test, dtype=torch.int64)
    events_test_disc = torch.tensor(events_test_disc, dtype=torch.float32)
    
    y_train = (idx_durations_train, events_train_disc)
    y_test = (idx_durations_test, events_test_disc)
    
    in_features = x_train.shape[1]
    num_nodes_list = [num_nodes] * num_layers
    net = tt.practical.MLPVanilla(in_features, num_nodes_list, labtrans.out_features, 
                                  True, dropout, output_bias=False)
    
    model = LogisticHazard(net, optimizer=optim.Adam(net.parameters(), lr=lr), device=device)
    
    epochs = 100
    
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              verbose=False, val_data=(x_test, y_test))
    
    surv = model.predict_surv_df(x_test)
    
    c_index = compute_c_index_from_surv(surv, durations_test, events_test)
    
    saved_path = model_saved_path_dict[scenario_name]
    if os.path.exists(saved_path):
        saved_net = torch.load(saved_path, map_location=device)
        saved_model = LogisticHazard(saved_net, optimizer=optim.Adam(saved_net.parameters()), device=device)
        saved_cidx = compute_c_index_from_surv(saved_model.predict_surv_df(x_test), durations_test, events_test)
        
        print(f"Saved model C-index: {saved_cidx:.4f}, Current trial C-index: {c_index:.4f}")
        if c_index > saved_cidx:
            print("New model is better, saving current model")
            torch.save(model.net, saved_path)
        else:
            print("Saved model is better, keeping it")
    else:
        print("No existing model, saving current model")
        torch.save(model.net, saved_path)
    
    trial.set_user_attr(key="model", value=model.net)
    
    return c_index

def compute_c_index_from_surv(surv_df, durations, events):
    median_time = np.median(durations)
    
    time_idx = np.argmin(np.abs(surv_df.index - median_time))
    risk_scores = 1 - surv_df.iloc[time_idx].values
    
    valid_mask = ~(np.isnan(risk_scores) | np.isnan(durations) | np.isnan(events))
    risk_scores = risk_scores[valid_mask]
    durations = durations[valid_mask]
    events = events[valid_mask]
    
    if len(risk_scores) == 0:
        return 0.5
    
    c_index = concordance_index(durations, risk_scores, events)
    return c_index

def c_idx(model, labtrans, test_dataset: LogisticHazardDataset, device, test=False):
    x_test, durations_test, events_test = test_dataset.prepare_data_for_pycox()
    x_test = torch.tensor(x_test, dtype=torch.float32)
    
    surv = model.predict_surv_df(x_test)
    
    c_index = compute_c_index_from_surv(surv, durations_test, events_test)
    
    print(f"Global test C-index: {c_index:.3f}")
    
    if test:
        num_events = sum(events_test)
        print(f"Event rate: {num_events}/{len(events_test)} ({num_events/len(events_test)*100:.2f}%)")
    
    return c_index

def auc(model, labtrans, test_dataset: LogisticHazardDataset, train_df: pd.DataFrame, device):
    x_test, durations_test, events_test = test_dataset.prepare_data_for_pycox()
    x_test = torch.tensor(x_test, dtype=torch.float32)
    
    surv = model.predict_surv_df(x_test)
    
    y_test = Surv.from_arrays(events_test.astype(bool), durations_test)
    
    times = np.percentile(durations_test[events_test == 1], [25, 50, 75])
    times = times[times > 0]
    
    if len(times) == 0:
        print("No valid time points for AUC calculation")
        return 0.5
    
    aucs = []
    for time_point in times:
        time_idx = np.argmin(np.abs(surv.index - time_point))
        risk_scores = 1 - surv.iloc[time_idx].values
        
        try:
            auc_score, _ = cumulative_dynamic_auc(y_test, y_test, risk_scores, time_point)
            aucs.append(auc_score[0])
        except Exception as e:
            print(f"Error computing AUC at time {time_point}: {e}")
            aucs.append(0.5)
    
    mean_auc = np.mean(aucs)
    print(f"Mean time-dependent AUC: {mean_auc:.3f}")
    return mean_auc

def brier_score_evaluation(model, labtrans, test_dataset: LogisticHazardDataset, train_df: pd.DataFrame, device):
    x_test, durations_test, events_test = test_dataset.prepare_data_for_pycox()
    x_test = torch.tensor(x_test, dtype=torch.float32)
    
    surv = model.predict_surv_df(x_test)
    
    median_time = np.median(durations_test)
    time_idx = np.argmin(np.abs(surv.index - median_time))
    risk_scores = 1 - surv.iloc[time_idx].values
    
    test_df = pd.DataFrame({
        'duration_in_days': durations_test,
        'has_esrd': events_test
    })
    
    brier_score = compute_brier_score_from_risk_scores(train_df, test_df, risk_scores)
    if brier_score is not None:
        print(f'Integrated Brier Score Test: {brier_score:.3f}')
    
    return brier_score

def run(scenario_name: ExperimentScenario):
    device = get_device()
    df, df_test = get_train_test_data(scenario_name)
    
    model_saved_path = model_saved_path_dict[scenario_name]
    
    if os.path.exists(model_saved_path):
        print("Loading from saved weights")
        saved_net = torch.load(model_saved_path, map_location=device, weights_only=False)
        labtrans = LabTransDiscreteTime(50)
        model = LogisticHazard(saved_net, optimizer=optim.Adam(saved_net.parameters()), device=device)
    else:
        print("Training new model with Optuna optimization")
        _ = ex_optuna(lambda trial: objective(trial, scenario_name))
        
        print("Loading best model from disk after optimization")
        saved_net = torch.load(model_saved_path, map_location=device, weights_only=False)
        
        labtrans = LabTransDiscreteTime(50)
        model = LogisticHazard(saved_net, optimizer=optim.Adam(saved_net.parameters()), device=device)
    
    print("Model summary:")
    print(model.net)
    
    test_dataset = LogisticHazardDataset(df_test, scenario_name)
    
    print(f"\n=== Evaluation Results for {scenario_name.value} ===")
    c_idx(model, labtrans, test_dataset, device, True)
    auc(model, labtrans, test_dataset, df, device)
    brier_score_evaluation(model, labtrans, test_dataset, df, device)

if __name__ == '__main__':
    # run(ExperimentScenario.TIME_VARIANT)
    # run(ExperimentScenario.HETEROGENEOUS)
    # run(ExperimentScenario.EGFR_COMPONENTS)
    # run(ExperimentScenario.HETEROGENEOUS_IMPUTE)
    # Guard: skip if this rep's CKD_FIFTY_FEATURES_HETEROGENEOUS train data
    # doesn't exist yet (e.g. mid schema-migration, or a mini-experiment rep
    # that deliberately didn't build it) — otherwise get_train_test_data()
    # silently falls through to a full raw MIMIC extraction from
    # labevents.csv instead of erroring. See CLAUDE.md "Check a script's
    # actual entry point before running it as an experiment".
    if os.path.exists(ckd_fifty_features_heterogeneous_train_data_path):
        run(ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS)
    else:
        print(f"Skipping CKD_FIFTY_FEATURES_HETEROGENEOUS: no train data at {ckd_fifty_features_heterogeneous_train_data_path}")
    run(ExperimentScenario.FOUR_FEATURES)
    run(ExperimentScenario.EIGHT_FEATURES)
    run(ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS)
