import joblib
import numpy as np
from lifelines.utils import concordance_index
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer

from pkgs.commons import egfr_ti_gbsa_model_path
from pkgs.data.model_data_store import get_train_test_data, sample
from pkgs.data.types import ExperimentScenario
from pkgs.experiments.utils import get_y_for_sckit_survival_model, round_metric, get_x_for_sckit_survival_model

def c_idx_score_fn(y, risk_score):
    events = np.array([item[0] for item in y])
    duration_in_days = np.array([item[1] for item in y])

    return concordance_index(duration_in_days, risk_score, events)

# non-time-variant model
def run_gbsa():
    df, df_test = get_train_test_data(ExperimentScenario.NON_TIME_VARIANT)
    df = sample(df)
    df_test = sample(df_test)
    
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
    joblib.dump(gbsa, egfr_ti_gbsa_model_path)
    
    print('Evaluate on test data')
    
    X_test = get_x_for_sckit_survival_model(df_test)
    risk_scores = gbsa.predict(X_test)
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
    run_gbsa()

