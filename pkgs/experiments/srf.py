import os
import datetime
import joblib
import numpy as np
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

from pkgs.commons import egfr_ti_srf_model_path
from pkgs.data.model_data_store import get_train_test_data
from pkgs.data.types import ExperimentScenario
from pkgs.experiments.utils import get_x_for_sckit_survival_model, get_y_for_sckit_survival_model, round_metric

# Data needs to be non-time-variant setup
# non-time-variant model
def run_survival_rf():
    df, df_test = get_train_test_data(ExperimentScenario.NON_TIME_VARIANT)

    X = get_x_for_sckit_survival_model(df)
    y = get_y_for_sckit_survival_model(df)

    if os.path.exists(egfr_ti_srf_model_path):
        rsf = joblib.load(egfr_ti_srf_model_path)
    else:
        print(f'Fitting Random Survival Forest model. Current time {datetime.datetime.now()}:\n')
        rsf = RandomSurvivalForest(n_jobs= 1, verbose=2, n_estimators=100)
        rsf.fit(X, y)
        joblib.dump(rsf, egfr_ti_srf_model_path)

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