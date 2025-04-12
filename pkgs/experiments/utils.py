import numpy as np
from lifelines.utils import concordance_index
from sksurv.metrics import integrated_brier_score
import torch
import optuna
from pkgs.data.types import ExperimentScenario
from torch.utils.data import Dataset
import pandas as pd

# from doc: "y must be a structured array with the first field being a binary class event indicator and the second field the time of the event/censoring"
def get_y_for_sckit_survival_model(df):
    return np.array(list(zip(df['has_esrd'].astype(bool), df['duration_in_days'])),
              dtype=[('event', bool), ('time', np.float64)])

# X must be a 2D array
def get_x_for_sckit_survival_model(df):
    X = df['egfr'].values.reshape(-1, 1)
    print(f'X shape: {X.shape}')
    return X


def round_metric(metric_num):
    return round(metric_num, 3)


def evaluate_ti_scikit_survival_model(df_test, risk_scores, surv_funcs, df_train):
    # Concordance Index on test data
    c_index_test = round_metric(concordance_index(df_test['duration_in_days'], risk_scores, df_test['has_esrd']))
    print(f'Concordance Index Test: {round_metric(c_index_test)}')
    
    # Brier score on test data
    times_test = np.linspace(0, df_test['duration_in_days'].max(), 100, endpoint=False)
    pred_surv_test = np.asarray([fn(times_test) for fn in surv_funcs])

    df_train['has_esrd'] = df_train['has_esrd'].astype(bool)
    df_test['has_esrd'] = df_test['has_esrd'].astype(bool)

    bs_test = integrated_brier_score(
        df_train[['has_esrd', 'duration_in_days']].to_records(index=False), 
        df_test[['has_esrd', 'duration_in_days']].to_records(index=False), pred_surv_test, times_test)
    print(f'Integrated Brier Score (Test): {round_metric(bs_test)}')

def c_idx_rnn_model(model, df_test, features):
    X_test = torch.tensor(df_test[features].values, dtype=torch.float32).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        test_risk_scores = model(X_test)
        test_risk_scores = test_risk_scores[:, -1, :]

    c_index = round_metric(concordance_index(df_test['duration_in_days'], test_risk_scores.squeeze().numpy(), df_test['has_esrd']))
    print("C-Index on Test Data:", c_index)

def ex_optuna(objective):
    print("Running Optuna hyperparameter optimization")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print(f"Best hyperparameters: {study.best_params}")
    best_model = trial.user_attrs["model"]

    return best_model

def get_tv_rnn_model_features(scenario_name: ExperimentScenario):
    if scenario_name == ExperimentScenario.TIME_VARIANT:
        return ['egfr']
    elif scenario_name == ExperimentScenario.HETEROGENEOUS:
        return ['egfr', 'egfr_missing', 'protein', 'protein_missing', 'albumin', 'albumin_missing']
    elif scenario_name == ExperimentScenario.EGFR_COMPONENTS:
        return ['age', 'gender', 'serum_creatinine']

def calculate_c_index(hazard_preds, time_intervals, event_indicators, num_risks):
    c_index_per_risk = []

    for risk in range(num_risks):
        risk_hazard_preds = hazard_preds[:, risk, :]
        risk_event_indicators = event_indicators[:, risk]
        observed_times = time_intervals[:, 0]

        # Cumulative hazard
        risk_scores = -torch.sum(torch.log(1 - risk_hazard_preds + 1e-8), dim=1).detach().cpu().numpy()
        mask = risk_event_indicators > 0

        print(observed_times[mask].detach().cpu().numpy())
        print(risk_scores[mask])
        print(risk_event_indicators[mask].detach().cpu().numpy())
        c_index = concordance_index(
                observed_times[mask].detach().cpu().numpy(),
                risk_scores[mask],
                event_observed=risk_event_indicators[mask].detach().cpu().numpy()
            )
        c_index_per_risk.append(c_index)

    return c_index_per_risk

class RNNAttentionDataset(Dataset):
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
                torch.FloatTensor(subject_data['has_esrd'].values))

def combine_loss(hazard_preds, time_intervals, event_indicators, num_risks, w1=0.5, w2=0.1):
    batch_size = hazard_preds.size(0)
    num_timepoints = hazard_preds.size(2)

    total_loss = 0

    for risk in range(num_risks):
        risk_hazard_preds = hazard_preds[:, risk, :]
        risk_event_indicators = event_indicators[:, risk]

        time_indices = time_intervals[:, 0].clamp(max=num_timepoints - 1).long()

        event_log_prob = torch.log(risk_hazard_preds[torch.arange(batch_size), time_indices]) * risk_event_indicators

        censor_log_prob = torch.zeros(batch_size, device=risk_hazard_preds.device)
        for i in range(batch_size):
            t = time_indices[i].item()
            if t > 0:
                censor_log_prob[i] = torch.sum(torch.log(1 - risk_hazard_preds[i, :t]))

        censor_log_prob = censor_log_prob * (1 - risk_event_indicators)

        log_likelihood_loss = -torch.mean(event_log_prob + censor_log_prob)

        ranking_loss = 0
        count = 0
        for i in range(batch_size):
            for j in range(batch_size):
                if time_intervals[i] < time_intervals[j] and risk_event_indicators[i] == 1:
                    t_i = time_indices[i].item()
                    F_i = torch.sum(risk_hazard_preds[i, :t_i])
                    F_j = torch.sum(risk_hazard_preds[j, :t_i])
                    ranking_loss += torch.exp(-(F_i - F_j) / w2)
                    count += 1

        if count > 0:
            ranking_loss /= count

        total_loss += log_likelihood_loss + w1 * ranking_loss

    return total_loss / num_risks
