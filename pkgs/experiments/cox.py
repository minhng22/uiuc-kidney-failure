#!/usr/bin/env python3
import os
import numpy as np
import joblib
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

from pkgs.commons import egfr_tv_cox_model_path, egfr_ti_cox_model_path
from pkgs.data.model_data_store import get_train_test_data
from pkgs.data.types import ExperimentScenario
from pkgs.experiments.utils import round_metric

def compute_time_dependent_auc(model, data_train, data_test, duration_col, event_col, times):
    y_train = Surv.from_dataframe(event=event_col, time=duration_col, data=data_train)
    y_test = Surv.from_dataframe(event=event_col, time=duration_col, data=data_test)
    risk_scores_test = model.predict_partial_hazard(data_test).values.flatten()
    auc_values, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores_test, times)
    return auc_values, mean_auc

def run_tv_cox_model():
    data_train, data_test = get_train_test_data(ExperimentScenario.TIME_VARIANT)

    if not os.path.exists(egfr_tv_cox_model_path):
        model = CoxTimeVaryingFitter()

        print(f'Fitting model:\n')
        model.fit(data_train, event_col='has_esrd', id_col='subject_id')

        joblib.dump(model, egfr_tv_cox_model_path)
    else:
        model = joblib.load(egfr_tv_cox_model_path)

    print('Evaluate on test data')

    risk_scores_test = model.predict_partial_hazard(data_test)
    c_index_test = round_metric(concordance_index(data_test['duration_in_days'], -risk_scores_test, data_test['has_esrd']))
    print(f'Concordance Index Test: {c_index_test}')

    last_obs = data_test.groupby('subject_id').last().reset_index()
    times = np.arange(30, 365, 30)
    train_last_obs = data_train.groupby('subject_id').last().reset_index()
    auc_values, mean_auc = compute_time_dependent_auc(model, train_last_obs, last_obs, 'stop', 'has_esrd', times)
    for t, auc in zip(times, auc_values):
        print(f"Time-dependent AUC at {t} days: {auc:.4f}")
    print(f"Mean time-dependent AUC: {mean_auc:.4f}")

def run_ti_cox_model():
    data_train, data_test = get_train_test_data(ExperimentScenario.TIME_INVARIANT)

    if not os.path.exists(egfr_ti_cox_model_path):
        model = CoxPHFitter()

        print(f'Fitting model:\n')
        model.fit(data_train, duration_col='duration_in_days', event_col='has_esrd')

        joblib.dump(model, egfr_ti_cox_model_path)
    else:
        model = joblib.load(egfr_ti_cox_model_path)

    print('Evaluate on test data')
    risk_scores_test = model.predict_partial_hazard(data_test)
    c_index_test = round_metric(concordance_index(data_test['duration_in_days'], -risk_scores_test, data_test['has_esrd']))
    print(f'Concordance Index Test: {c_index_test}')

    times = np.arange(30, 365, 30)
    auc_values, mean_auc = compute_time_dependent_auc(model, data_train, data_test, 'duration_in_days', 'has_esrd', times)
    for t, auc in zip(times, auc_values):
        print(f"Time-dependent AUC at {t} days: {auc:.4f}")
    print(f"Mean time-dependent AUC: {mean_auc:.4f}")

if __name__ == "__main__":
    print("\nRunning time-variant Cox model evaluation with time-dependent AUC...")
    run_tv_cox_model()
