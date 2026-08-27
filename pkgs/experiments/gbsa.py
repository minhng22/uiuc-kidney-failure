import os
import joblib
import numpy as np
from lifelines.utils import concordance_index
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer

from pkgs.commons import (
    egfr_ti_gbsa_model_path, four_features_gbsa_model_path,
    eight_features_gbsa_model_path, twenty_features_heterogeneous_gbsa_model_path,
)
from pkgs.data_analysis.model_data_store import get_train_test_data, get_last_observation_data
from pkgs.data_analysis.types import ExperimentScenario
from pkgs.experiments.utils import get_y_for_sckit_survival_model, round_metric, get_x_for_sckit_survival_model, load_pkl_and_dill_model, compute_brier_score_from_risk_scores
import dill

# Path dict + scenario-aware run added for Stage 3's four/eight/
# twenty_features_heterogeneous scenarios, alongside (not replacing) the
# original NON_TIME_VARIANT-only run_gbsa() above. These 3 scenarios are
# time-varying (many rows per subject_id); get_last_observation_data()
# flattens each subject to their single last (most recent) observation,
# since GBSA (like srf/survival_svm/weibul/deepsurv) has no per-subject-
# sequence notion the way cox/ddh/hazard_transformer/logistic_hazard/
# rnn_surv do.
gbsa_model_path_dict = {
    ExperimentScenario.FOUR_FEATURES: four_features_gbsa_model_path,
    ExperimentScenario.EIGHT_FEATURES: eight_features_gbsa_model_path,
    ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS: twenty_features_heterogeneous_gbsa_model_path,
}

def c_idx_score_fn(y, risk_score):
    events = np.array([item[0] for item in y])
    duration_in_days = np.array([item[1] for item in y])
    return concordance_index(duration_in_days, risk_score, events)

def evaluate_model(gbsa, df, df_test):
    print('Evaluate on test data')
    
    X_test = get_x_for_sckit_survival_model(df_test)
    risk_scores = gbsa.predict(X_test)
    times = np.arange(1, 730, 1)

    print(f'Risk scores shape: {risk_scores.shape}')
    print(f'First 10 risk scores: {risk_scores[:10]}')
    
    # Concordance Index on test data
    c_index_test = round_metric(concordance_index(df_test['duration_in_days'], risk_scores, df_test['has_esrd']))
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

# non-time-variant model
def run_gbsa():
    df, df_test = get_train_test_data(ExperimentScenario.NON_TIME_VARIANT)

    trained_model = load_pkl_and_dill_model(egfr_ti_gbsa_model_path)
    
    if trained_model:
        print(f'Model found at {egfr_ti_gbsa_model_path}. Loading model...')
        gbsa = trained_model
    else:
        print('No existing model found. Starting hyperparameter tuning...')
        df, df_test = get_train_test_data(ExperimentScenario.NON_TIME_VARIANT)
    
        X = get_x_for_sckit_survival_model(df)
        y = get_y_for_sckit_survival_model(df)
    
        print(df.head())
    
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 10, 15],
            'learning_rate': [0.01, 0.1, 0.2],
            'min_samples_split': [2, 5, 10, 15],
        }
    
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
        scorer = make_scorer(c_idx_score_fn, greater_is_better=True)
    
        grid_search = GridSearchCV(
            estimator=GradientBoostingSurvivalAnalysis(verbose=0),
            param_grid=param_grid,
            scoring=scorer,
            cv=cv,
            n_jobs=1,
            verbose=2,
        )
    
        print('Starting hyperparameter tuning...')
        grid_search.fit(X, y)
    
        print('Best parameters found:')
        print(grid_search.best_params_)
    
        gbsa = grid_search.best_estimator_
        with open(egfr_ti_gbsa_model_path, 'wb') as f:
            dill.dump(gbsa, f, protocol=4)
        
    print('Evaluate on test data')
    
    X_test = get_x_for_sckit_survival_model(df_test)
    risk_scores = gbsa.predict(X_test)
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
    twenty_features_heterogeneous. See gbsa_model_path_dict's comment above
    for why get_last_observation_data() is used instead of the raw
    time-varying data."""
    df, df_test = get_last_observation_data(scenario)
    model_path = gbsa_model_path_dict[scenario]

    trained_model = load_pkl_and_dill_model(model_path)

    if trained_model:
        print(f'Model found at {model_path}. Loading model...')
        gbsa = trained_model
    else:
        print('No existing model found. Starting hyperparameter tuning...')

        X = get_x_for_sckit_survival_model(df, scenario)
        y = get_y_for_sckit_survival_model(df)

        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 10, 15],
            'learning_rate': [0.01, 0.1, 0.2],
            'min_samples_split': [2, 5, 10, 15],
        }

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(c_idx_score_fn, greater_is_better=True)

        grid_search = GridSearchCV(
            estimator=GradientBoostingSurvivalAnalysis(verbose=0),
            param_grid=param_grid,
            scoring=scorer,
            cv=cv,
            n_jobs=1,
            verbose=2,
        )

        print('Starting hyperparameter tuning...')
        grid_search.fit(X, y)

        print('Best parameters found:')
        print(grid_search.best_params_)

        gbsa = grid_search.best_estimator_
        with open(model_path, 'wb') as f:
            dill.dump(gbsa, f, protocol=4)

    X_test = get_x_for_sckit_survival_model(df_test, scenario)
    risk_scores = gbsa.predict(X_test)

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


def joblib_to_dill():
    model_path = egfr_ti_gbsa_model_path
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        with open(model_path.replace('.pkl', '.dill'), 'wb') as f:
            dill.dump(model, f, protocol=4)

if __name__ == '__main__':
    run_gbsa()
    print("\nRunning FOUR_FEATURES GBSA model evaluation...")
    run_scenario(ExperimentScenario.FOUR_FEATURES)
    print("\nRunning EIGHT_FEATURES GBSA model evaluation...")
    run_scenario(ExperimentScenario.EIGHT_FEATURES)
    print("\nRunning TWENTY_FEATURES_HETEROGENEOUS GBSA model evaluation...")
    run_scenario(ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS)

