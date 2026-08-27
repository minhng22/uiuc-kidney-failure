import os
import datetime
import joblib
import numpy as np
import dill
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer

from pkgs.commons import (
    egfr_ti_srf_model_path, four_features_srf_model_path,
    eight_features_srf_model_path, twenty_features_heterogeneous_srf_model_path,
)
from pkgs.data_analysis.model_data_store import get_train_test_data, sample, get_last_observation_data
from pkgs.data_analysis.types import ExperimentScenario
from pkgs.experiments.utils import get_x_for_sckit_survival_model, get_y_for_sckit_survival_model, round_metric, load_pkl_and_dill_model, compute_brier_score_from_risk_scores

# Path dict + scenario-aware run added for Stage 3's four/eight/
# twenty_features_heterogeneous scenarios, alongside (not replacing) the
# original NON_TIME_VARIANT-only run_survival_rf() below. See gbsa.py's
# equivalent comment for why get_last_observation_data() is used.
srf_model_path_dict = {
    ExperimentScenario.FOUR_FEATURES: four_features_srf_model_path,
    ExperimentScenario.EIGHT_FEATURES: eight_features_srf_model_path,
    ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS: twenty_features_heterogeneous_srf_model_path,
}

def c_idx_score_fn(y, risk_score):
    events = np.array([item[0] for item in y])
    duration_in_days = np.array([item[1] for item in y])
    return concordance_index(duration_in_days, risk_score, events)

# Data needs to be non-time-variant setup
# non-time-variant model
def run_survival_rf():
    df, df_test = get_train_test_data(ExperimentScenario.NON_TIME_VARIANT)

    trained_model = load_pkl_and_dill_model(egfr_ti_srf_model_path)

    if trained_model:
        print(f'Model file found at {egfr_ti_srf_model_path}. Loading model...')
        rsf = trained_model
        evaluate_model(rsf, df, df_test)
    else:
        print(f'No existing model found. Training model. Current time {datetime.datetime.now()}:\n')
                
        X = get_x_for_sckit_survival_model(df)
        y = get_y_for_sckit_survival_model(df)
        
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10, 15]
        }
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(c_idx_score_fn, greater_is_better=True)
        
        grid_search = GridSearchCV(
            estimator=RandomSurvivalForest(n_jobs=2, verbose=2),
            param_grid=param_grid,
            scoring=scorer,
            cv=cv,
            verbose=2,
        )
        
        grid_search.fit(X, y)
        
        print('Best parameters found:')
        print(grid_search.best_params_)
        
        rsf = grid_search.best_estimator_
        with open(egfr_ti_srf_model_path, 'wb') as f:
            dill.dump(rsf, f, protocol=4)
        print(f'Model saved to {egfr_ti_srf_model_path}')
        
    evaluate_model(rsf, df, df_test)

def evaluate_model(rsf, df, df_test):
    print('Evaluate on test data')
    
    X_test = get_x_for_sckit_survival_model(df_test)
    risk_scores = rsf.predict(X_test)
    times = np.arange(1, 730, 1)
    
    print(f'Risk scores shape: {risk_scores.shape}')
    print(f'First 10 risk scores: {risk_scores[:10]}')
    
    # Concordance Index on test data
    c_index_test = round_metric(concordance_index(df_test['duration_in_days'], 1 - risk_scores, df_test['has_esrd']))
    print(f'Concordance Index Test: {round_metric(c_index_test)}')
    
    # Compute time-dependent AUC
    y_train = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df)
    y_test = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df_test)
    _, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)
    print(f'Mean AUC: {round_metric(mean_auc)}')
    
    # Compute Brier Score
    brier_score = compute_brier_score_from_risk_scores(df, df_test, risk_scores)
    if brier_score is not None:
        print(f'Integrated Brier Score Test: {brier_score}')

def run_scenario(scenario: ExperimentScenario):
    """Scenario-aware entry point for four_features/eight_features/
    twenty_features_heterogeneous."""
    df, df_test = get_last_observation_data(scenario)
    model_path = srf_model_path_dict[scenario]

    trained_model = load_pkl_and_dill_model(model_path)

    if trained_model:
        print(f'Model file found at {model_path}. Loading model...')
        rsf = trained_model
    else:
        print(f'No existing model found. Training model. Current time {datetime.datetime.now()}:\n')

        X = get_x_for_sckit_survival_model(df, scenario)
        y = get_y_for_sckit_survival_model(df)

        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10, 15]
        }

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(c_idx_score_fn, greater_is_better=True)

        grid_search = GridSearchCV(
            estimator=RandomSurvivalForest(n_jobs=2, verbose=2),
            param_grid=param_grid,
            scoring=scorer,
            cv=cv,
            verbose=2,
        )

        grid_search.fit(X, y)

        print('Best parameters found:')
        print(grid_search.best_params_)

        rsf = grid_search.best_estimator_
        with open(model_path, 'wb') as f:
            dill.dump(rsf, f, protocol=4)
        print(f'Model saved to {model_path}')

    X_test = get_x_for_sckit_survival_model(df_test, scenario)
    risk_scores = rsf.predict(X_test)

    c_index_test = round_metric(concordance_index(df_test['duration_in_days'], 1 - risk_scores, df_test['has_esrd']))
    print(f'Concordance Index Test: {c_index_test}')

    y_train = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df)
    y_test = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df_test)
    times = np.arange(1, min(730, int(df_test['duration_in_days'].max())), 1)
    try:
        _, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)
        print(f'Mean AUC: {round_metric(mean_auc)}')
    except Exception as e:
        print(f"Warning: could not compute AUC: {e}")

    brier_score = compute_brier_score_from_risk_scores(df, df_test, risk_scores)
    if brier_score is not None:
        print(f'Integrated Brier Score Test: {brier_score}')


if __name__ == '__main__':
    run_survival_rf()
    print("\nRunning FOUR_FEATURES Survival RF model evaluation...")
    run_scenario(ExperimentScenario.FOUR_FEATURES)
    print("\nRunning EIGHT_FEATURES Survival RF model evaluation...")
    run_scenario(ExperimentScenario.EIGHT_FEATURES)
    print("\nRunning TWENTY_FEATURES_HETEROGENEOUS Survival RF model evaluation...")
    run_scenario(ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS)