import os
import joblib
import numpy as np
from lifelines.utils import concordance_index
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

from pkgs.commons import egfr_ti_gbsa_model_path
from pkgs.data.model_data_store import get_train_test_data
from pkgs.data.types import ExperimentScenario
from pkgs.experiments.utils import get_y_for_sckit_survival_model, round_metric, get_x_for_sckit_survival_model

# non-time-variant model
def run_gbsa():
    df, df_test = get_train_test_data(ExperimentScenario.NON_TIME_VARIANT)
    
    X = get_x_for_sckit_survival_model(df)
    y = get_y_for_sckit_survival_model(df)
    
    print(df.head())
    
    if os.path.exists(egfr_ti_gbsa_model_path):
        gbsa = joblib.load(egfr_ti_gbsa_model_path)
    else:
        print('Fitting Gradient Boosting Survival Analysis model')
        gbsa = GradientBoostingSurvivalAnalysis(verbose=2, n_estimators=100, max_depth=10)
        gbsa.fit(X, y)
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

