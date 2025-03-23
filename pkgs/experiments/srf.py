from pkgs.commons import egfr_tv_srf_model_path
from pkgs.data.model_data_store import get_train_test_data_egfr
from pkgs.experiments.utils import get_y, evaluate_sc_and_cox_survival

import joblib
from sksurv.ensemble import RandomSurvivalForest

import datetime
import os

# Data needs to be time-invariant setup
def run_survival_rf():
    df, df_test = get_train_test_data_egfr(True)
    df['has_esrd'] = df['has_esrd'].astype(bool)

    X = df[['duration_in_days', 'egfr']]
    y = get_y(df)

    if os.path.exists(egfr_tv_srf_model_path):
        rsf = joblib.load(egfr_tv_srf_model_path)
    else:
        print(f'Fitting Random Survival Forest model. Current time {datetime.datetime.now()}:\n')
        rsf = RandomSurvivalForest(n_jobs= 2, verbose=2)
        rsf.fit(X, y)
        joblib.dump(rsf, egfr_tv_srf_model_path)

    df_test['has_esrd'] = df_test['has_esrd'].astype(bool)
    X_test = df_test[['duration_in_days', 'egfr']]

    evaluate_sc_and_cox_survival(df_test, -rsf.predict(X_test), rsf.predict_survival_function(X_test), y)


if __name__ == '__main__':
    run_survival_rf()