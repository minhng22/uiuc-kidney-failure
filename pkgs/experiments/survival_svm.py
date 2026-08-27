import os
import numpy as np
import joblib
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv
import pandas as pd

from pkgs.commons import (
    egfr_ti_survival_svm_model_path, four_features_survival_svm_model_path,
    eight_features_survival_svm_model_path, twenty_features_heterogeneous_survival_svm_model_path,
)
from pkgs.data_analysis.model_data_store import get_train_test_data, get_last_observation_data
from pkgs.data_analysis.types import ExperimentScenario
from pkgs.experiments.utils import round_metric, load_pkl_and_dill_model, get_tv_rnn_model_features
import dill

# Path dict + scenario-aware run added for Stage 3's four/eight/
# twenty_features_heterogeneous scenarios, alongside (not replacing) the
# original NON_TIME_VARIANT-only run_ti_survival_svm_model() below. See
# gbsa.py's equivalent comment for why get_last_observation_data() is used.
# Unlike run_ti_survival_svm_model()'s generic "every column except
# duration/event/subject_id/Unnamed: 0" feature selection (fine when
# NON_TIME_VARIANT's data has nothing else in it), these 3 scenarios' data
# also carries start/stop columns from the time-varying source, so features
# are selected explicitly via get_tv_rnn_model_features() instead.
survival_svm_model_path_dict = {
    ExperimentScenario.FOUR_FEATURES: four_features_survival_svm_model_path,
    ExperimentScenario.EIGHT_FEATURES: eight_features_survival_svm_model_path,
    ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS: twenty_features_heterogeneous_survival_svm_model_path,
}

def compute_time_dependent_auc(model, data_train, data_test, duration_col, event_col, times):
    """Compute time-dependent AUC for Survival SVM"""
    # Filter training data
    valid_mask_train = data_train[duration_col] > 0
    data_train_filtered = data_train[valid_mask_train].copy()
    
    y_train = Surv.from_dataframe(event=event_col, time=duration_col, data=data_train_filtered)
    y_test = Surv.from_dataframe(event=event_col, time=duration_col, data=data_test)
    
    # Get features (exclude time and event columns)
    feature_cols = [col for col in data_test.columns 
                   if col not in [duration_col, event_col, 'subject_id', 'Unnamed: 0']]
    X_test = data_test[feature_cols].values
    
    # Get risk scores (negative because higher risk score means worse survival)
    risk_scores = -model.predict(X_test)
    
    print(f"Risk scores test: {risk_scores.shape}")
    auc_values, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)
    return auc_values, mean_auc

def compute_brier_score(model, data_train, data_test, duration_col, event_col, times):
    """Compute integrated Brier score for Survival SVM"""
    try:
        from sksurv.nonparametric import kaplan_meier_estimator
        
        # Filter training data
        valid_mask_train = data_train[duration_col] > 0
        data_train_filtered = data_train[valid_mask_train].copy()
        
        y_train = Surv.from_dataframe(event=event_col, time=duration_col, data=data_train_filtered)
        y_test = Surv.from_dataframe(event=event_col, time=duration_col, data=data_test)
        
        # Get features
        feature_cols = [col for col in data_test.columns 
                       if col not in [duration_col, event_col, 'subject_id', 'Unnamed: 0']]
        X_test = data_test[feature_cols].values
        
        # Get risk scores
        risk_scores = model.predict(X_test)
        
        # For Survival SVM, we use the Kaplan-Meier estimator on training data
        # as a baseline survival function, then adjust by risk scores
        time_train, survival_prob_train = kaplan_meier_estimator(
            y_train["has_esrd"], y_train["duration_in_days"]
        )
        
        # Interpolate survival probabilities at desired time points
        from scipy.interpolate import interp1d
        # Extend survival function to cover all time points
        extended_times = np.concatenate([[0], time_train, [times.max() + 1]])
        extended_probs = np.concatenate([[1.0], survival_prob_train, [survival_prob_train[-1]]])
        
        interp_func = interp1d(extended_times, extended_probs, 
                              kind='previous', bounds_error=False, fill_value=0.0)
        baseline_survival = interp_func(times)
        
        # Adjust survival probabilities based on risk scores
        # Higher risk score means lower survival probability
        risk_scores_normalized = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min() + 1e-8)
        
        # Create survival probability matrix: [n_samples, n_times]
        survival_probs = np.outer(1 - risk_scores_normalized, 1 - baseline_survival) + np.outer(risk_scores_normalized, baseline_survival)
        survival_probs = np.clip(survival_probs, 0.0, 1.0)
        
        brier_score = integrated_brier_score(y_train, y_test, survival_probs, times)
        return round_metric(brier_score)
    except Exception as e:
        print(f"Warning: Could not compute Brier score: {e}")
        return None

def run_ti_survival_svm_model():
    """Run Survival SVM model for non-time-variant scenario"""
    data_train, data_test = get_train_test_data(ExperimentScenario.NON_TIME_VARIANT)
    
    print(f"Train data path {data_train.attrs.get('path', 'N/A')}")
    print(f"Test data path {data_test.attrs.get('path', 'N/A')}")
    print(f"Number of patients: test {len(data_test)} and train {len(data_train)}")

    model_path = egfr_ti_survival_svm_model_path

    trained_model = load_pkl_and_dill_model(model_path)

    if not trained_model:
        # Filter out invalid time values (<=0) which scikit-survival doesn't allow
        valid_mask_train = data_train['duration_in_days'] > 0
        data_train_filtered = data_train[valid_mask_train].copy()
        
        print(f"Filtered out {len(data_train) - len(data_train_filtered)} training samples with duration <= 0")
        
        # Prepare data for Survival SVM
        feature_cols = [col for col in data_train_filtered.columns 
                       if col not in ['duration_in_days', 'has_esrd', 'subject_id', 'Unnamed: 0']]
        X_train = data_train_filtered[feature_cols].values
        y_train = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=data_train_filtered)

        print(f'Fitting Survival SVM model with {len(feature_cols)} features...')
        print(f'Feature columns: {feature_cols}')
        
        model = FastSurvivalSVM(alpha=0.01, max_iter=1000, tol=1e-6, random_state=42)
        model.fit(X_train, y_train)

        with open(model_path, 'wb') as f:
            dill.dump(model, f, protocol=4)
    else:
        model = trained_model

    print('Evaluate on test data')

    # Filter out invalid time values from test data too
    valid_mask_test = data_test['duration_in_days'] > 0
    data_test_filtered = data_test[valid_mask_test].copy()
    
    print(f"Filtered out {len(data_test) - len(data_test_filtered)} test samples with duration <= 0")
    
    # Prepare test data
    feature_cols = [col for col in data_test_filtered.columns 
                   if col not in ['duration_in_days', 'has_esrd', 'subject_id', 'Unnamed: 0']]
    X_test = data_test_filtered[feature_cols].values
    y_test = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=data_test_filtered)

    # Predict risk scores
    risk_scores = model.predict(X_test)
    print(f"Risk scores shape: {risk_scores.shape}")
    print(f"First 10 risk scores: {risk_scores[:10]}")
    
    # Compute C-index
    c_index_test = concordance_index_censored(
        data_test_filtered['has_esrd'].astype(bool), 
        data_test_filtered['duration_in_days'], 
        risk_scores
    )[0]
    print(f'Concordance Index Test: {round_metric(c_index_test)}')

    # Compute Brier Score
    times = np.arange(1, min(730, data_test_filtered['duration_in_days'].max()), 1)
    brier_score = compute_brier_score(model, data_train, data_test_filtered, 'duration_in_days', 'has_esrd', times)
    if brier_score is not None:
        print(f'Integrated Brier Score Test: {brier_score}')

    # Compute time-dependent AUC  
    _, mean_auc = compute_time_dependent_auc(model, data_train, data_test_filtered, 'duration_in_days', 'has_esrd', times)
    print(f"Mean AUC: {mean_auc:.3f}")

def run_scenario(scenario: ExperimentScenario):
    """Scenario-aware entry point for four_features/eight_features/
    twenty_features_heterogeneous."""
    data_train, data_test = get_last_observation_data(scenario)
    features = get_tv_rnn_model_features(scenario)
    cols = features + ['duration_in_days', 'has_esrd']
    data_train = data_train[cols].copy()
    data_test = data_test[cols].copy()

    model_path = survival_svm_model_path_dict[scenario]
    trained_model = load_pkl_and_dill_model(model_path)

    if not trained_model:
        valid_mask_train = data_train['duration_in_days'] > 0
        data_train_filtered = data_train[valid_mask_train].copy()
        print(f"Filtered out {len(data_train) - len(data_train_filtered)} training samples with duration <= 0")

        X_train = data_train_filtered[features].values
        y_train = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=data_train_filtered)

        print(f'Fitting Survival SVM model with {len(features)} features: {features}')
        model = FastSurvivalSVM(alpha=0.01, max_iter=1000, tol=1e-6, random_state=42)
        model.fit(X_train, y_train)

        with open(model_path, 'wb') as f:
            dill.dump(model, f, protocol=4)
    else:
        model = trained_model

    valid_mask_test = data_test['duration_in_days'] > 0
    data_test_filtered = data_test[valid_mask_test].copy()
    print(f"Filtered out {len(data_test) - len(data_test_filtered)} test samples with duration <= 0")

    X_test = data_test_filtered[features].values
    risk_scores = model.predict(X_test)

    c_index_test = concordance_index_censored(
        data_test_filtered['has_esrd'].astype(bool),
        data_test_filtered['duration_in_days'],
        risk_scores
    )[0]
    print(f'Concordance Index Test: {round_metric(c_index_test)}')

    times = np.arange(1, min(730, int(data_test_filtered['duration_in_days'].max())), 1)
    try:
        brier_score = compute_brier_score(model, data_train, data_test_filtered, 'duration_in_days', 'has_esrd', times)
        if brier_score is not None:
            print(f'Integrated Brier Score Test: {brier_score}')
    except Exception as e:
        print(f"Warning: could not compute Brier score: {e}")

    try:
        _, mean_auc = compute_time_dependent_auc(model, data_train, data_test_filtered, 'duration_in_days', 'has_esrd', times)
        print(f"Mean AUC: {mean_auc:.3f}")
    except Exception as e:
        print(f"Warning: could not compute AUC: {e}")


def run_all():
    """Run all Survival SVM experiments"""
    print("\nRunning non-time-variant Survival SVM model evaluation...")
    run_ti_survival_svm_model()

    print("\nTime-variant scenarios not supported for Survival SVM")
    print("Skipping TIME_VARIANT, HETEROGENEOUS, and EGFR_COMPONENTS scenarios")

if __name__ == "__main__":
    run_all()
