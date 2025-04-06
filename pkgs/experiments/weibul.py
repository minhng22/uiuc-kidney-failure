import os
import joblib
import numpy as np
from lifelines import WeibullAFTFitter
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

from pkgs.commons import egfr_ti_weibul_model_path
from pkgs.data.model_data_store import get_train_test_data
from pkgs.data.types import ExperimentScenario

def compute_time_dependent_auc(model: WeibullAFTFitter, data_train, data_test, duration_col, event_col, times):
    y_train = Surv.from_dataframe(event=event_col, time=duration_col, data=data_train)
    y_test = Surv.from_dataframe(event=event_col, time=duration_col, data=data_test)
    cum_hazard = model.predict_cumulative_hazard(data_test, times=times).values
    cum_hazard = cum_hazard.T # shape of weibull model is (n_times, n_samples). shapes required for cumulative_dynamic_auc is (n_samples, n_times) 

    print(f"cumulative hazard shape: {cum_hazard.shape}")
    auc_values, mean_auc = cumulative_dynamic_auc(y_train, y_test, cum_hazard, times)

    return auc_values, mean_auc

def run_ti():
    df, df_test = get_train_test_data(ExperimentScenario.NON_TIME_VARIANT)

    df['duration_in_days'] = df['duration_in_days'].replace(0, 1e-5)
    df_test['duration_in_days'] = df_test['duration_in_days'].replace(0, 1e-5)

    print(f"Train data shape: {df.shape}")
    print(f"Test data shape: {df_test.shape}")

    if not os.path.exists(egfr_ti_weibul_model_path):
        model = WeibullAFTFitter()
        print('Fitting model:')
        model.fit(df, event_col='has_esrd', duration_col='duration_in_days')
        joblib.dump(model, egfr_ti_weibul_model_path)
    else:
        print("Loading model from file")
        model = joblib.load(egfr_ti_weibul_model_path)

    print('Evaluate on test data')
    times = np.arange(1, 365, 1)

    predicted_survival_times = model.predict_median(df_test)
    event_occurred = df_test['has_esrd'].values
    actual_survival_times = df_test['duration_in_days'].values

    c_index = concordance_index(actual_survival_times, predicted_survival_times, event_occurred)
    print(f"C-index: {c_index:.4f}")

    _, mean_auc = compute_time_dependent_auc(model, df, df_test, 'duration_in_days', 'has_esrd', times)
    print(f"Mean time-dependent AUC: {mean_auc:.4f}")

if __name__ == '__main__':
    run_ti()