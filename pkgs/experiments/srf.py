from pkgs.commons import egfr_ti_srf_model_path
from pkgs.data.model_data_store import get_train_test_data
from pkgs.experiments.utils import get_y_for_sckit_survival_model, evaluate_ti_scikit_survival_model

import joblib
from sksurv.ensemble import RandomSurvivalForest

import datetime
import os

from pkgs.data.types import ExperimentScenario

# Data needs to be time-invariant setup
# time-invariant model
def run_survival_rf():
    df, df_test = get_train_test_data(ExperimentScenario.TIME_INVARIANT)
    df['has_esrd'] = df['has_esrd'].astype(bool)

    X = df[['start', 'stop', 'egfr']]
    y = get_y_for_sckit_survival_model(df)

    if os.path.exists(egfr_ti_srf_model_path):
        rsf = joblib.load(egfr_ti_srf_model_path)
    else:
        print(f'Fitting Random Survival Forest model. Current time {datetime.datetime.now()}:\n')
        rsf = RandomSurvivalForest(n_jobs= 1, verbose=2, n_estimators=100)
        rsf.fit(X, y)
        joblib.dump(rsf, egfr_ti_srf_model_path)

    df_test['has_esrd'] = df_test['has_esrd'].astype(bool)
    X_test = df_test[['start', 'stop', 'egfr']]

    evaluate_ti_scikit_survival_model(df_test, -rsf.predict(X_test), rsf.predict_survival_function(X_test), df)


if __name__ == '__main__':
    run_survival_rf()