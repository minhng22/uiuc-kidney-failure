import os
import datetime
import joblib
import numpy as np
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer

from pkgs.commons import egfr_ti_srf_model_path
from pkgs.data.model_data_store import get_train_test_data, sample
from pkgs.data.types import ExperimentScenario
from pkgs.experiments.utils import get_x_for_sckit_survival_model, get_y_for_sckit_survival_model, round_metric

def c_idx_score_fn(y, risk_score):
    events = np.array([item[0] for item in y])
    duration_in_days = np.array([item[1] for item in y])
    return concordance_index(duration_in_days, risk_score, events)

# Data needs to be non-time-variant setup
# non-time-variant model
def run_survival_rf():
    df, df_test = get_train_test_data(ExperimentScenario.NON_TIME_VARIANT)

    if os.path.exists(egfr_ti_srf_model_path):
        print(f'Model file found at {egfr_ti_srf_model_path}. Loading model...')
        rsf = joblib.load(egfr_ti_srf_model_path)
        evaluate_model(rsf, df, df_test)
    else:
        print(f'No existing model found. Training model. Current time {datetime.datetime.now()}:\n')
                
        X = get_x_for_sckit_survival_model(df)
        y = get_y_for_sckit_survival_model(df)
        
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10, 15]
        }
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(c_idx_score_fn, greater_is_better=True)
        
        grid_search = GridSearchCV(
            estimator=RandomSurvivalForest(n_jobs=1, verbose=0),
            param_grid=param_grid,
            scoring=scorer,
            cv=cv,
            n_jobs=1
        )
        
        grid_search.fit(X, y)
        
        print('Best parameters found:')
        print(grid_search.best_params_)
        
        rsf = grid_search.best_estimator_
        joblib.dump(rsf, egfr_ti_srf_model_path)
        print(f'Model saved to {egfr_ti_srf_model_path}')
        
        evaluate_model(rsf, df, df_test)

def evaluate_model(rsf, df, df_test):
    print('Evaluate on test data')
    
    X_test = get_x_for_sckit_survival_model(df_test)
    risk_scores = rsf.predict(X_test)
    times = np.arange(1, 365, 1)
    
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

if __name__ == '__main__':
    run_survival_rf()