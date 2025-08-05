import numpy as np
from lifelines.utils import concordance_index
from sksurv.metrics import integrated_brier_score, brier_score
from sksurv.util import Surv
import torch
import optuna
from pkgs.data.types import ExperimentScenario
import os
import dill
import joblib

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


def round_metric(metric_num):
    return round(metric_num, 3)


def compute_brier_score_from_survival_probs(df_train, df_test, survival_probs, times):
    """
    Compute Integrated Brier Score from survival probabilities.
    
    Args:
        df_train: Training dataframe with 'has_esrd' and 'duration_in_days' columns
        df_test: Test dataframe with 'has_esrd' and 'duration_in_days' columns  
        survival_probs: Array of survival probabilities of shape (n_samples, n_times)
        times: Array of time points corresponding to survival probabilities
    
    Returns:
        Integrated Brier Score
    """
    y_train = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df_train)
    y_test = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df_test)
    
    try:
        # Ensure survival_probs has the right shape (n_samples, n_times)
        if survival_probs.ndim == 1:
            survival_probs = survival_probs.reshape(1, -1)
        elif survival_probs.shape[0] != len(df_test):
            survival_probs = survival_probs.T
            
        ibs = integrated_brier_score(y_train, y_test, survival_probs, times)
        return round_metric(ibs)
    except Exception as e:
        print(f"Warning: Could not compute Brier Score: {e}")
        return None


def compute_brier_score_from_risk_scores(df_train, df_test, risk_scores):
    """
    Compute Brier Score from risk scores by converting to survival probabilities.
    Uses a simple exponential survival model approximation.
    
    Args:
        df_train: Training dataframe
        df_test: Test dataframe  
        risk_scores: Risk scores from the model
    
    Returns:
        Integrated Brier Score
    """
    try:
        # Define time points for evaluation
        max_time = max(df_train['duration_in_days'].max(), df_test['duration_in_days'].max())
        times = np.linspace(1, min(max_time, 365), 50)  # Evaluate up to 1 year
        
        # Convert risk scores to survival probabilities using exponential model
        # S(t) = exp(-lambda * t), where lambda is proportional to risk score
        # Normalize risk scores to be positive
        risk_scores_norm = risk_scores - risk_scores.min() + 0.01
        
        # Create survival probability matrix
        survival_probs = np.zeros((len(df_test), len(times)))
        for i, t in enumerate(times):
            survival_probs[:, i] = np.exp(-risk_scores_norm * t / 365.0)  # Scale by year
            
        return compute_brier_score_from_survival_probs(df_train, df_test, survival_probs, times)
    except Exception as e:
        print(f"Warning: Could not compute Brier Score from risk scores: {e}")
        return None


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
    
    # Also compute Brier Score using our new comprehensive function
    brier_score_new = compute_brier_score_from_survival_probs(df_train, df_test, pred_surv_test.T, times_test)
    if brier_score_new is not None:
        print(f'Integrated Brier Score (New): {brier_score_new}')

def c_idx_rnn_model(model, df_test, features):
    X_test = torch.tensor(df_test[features].values, dtype=torch.float32).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        test_risk_scores = model(X_test)
        test_risk_scores = test_risk_scores[:, -1, :]

    c_index = round_metric(concordance_index(df_test['duration_in_days'], test_risk_scores.squeeze().numpy(), df_test['has_esrd']))
    print("C-Index on Test Data:", c_index)

def ex_optuna(objective, n_trials=10):
    print("Running Optuna hyperparameter optimization")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

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

        total_loss += log_likelihood_loss * w1 + ranking_loss * w2

    return total_loss / num_risks

def load_pkl_and_dill_model(model_path):
    model_pkl_path = model_path.replace('.dill', '.pkl')

    if not os.path.exists(model_path) and not os.path.exists(model_pkl_path):
        return None
    
    # some old models saved with .pkl extension
    if os.path.exists(model_pkl_path):
        print(f"Loading model from {model_pkl_path}")
        return joblib.load(model_pkl_path)
    
    print(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        return dill.load(f)
    