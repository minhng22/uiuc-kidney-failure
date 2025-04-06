import os
import numpy as np
import joblib
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

from pkgs.commons import egfr_tv_cox_model_path, egfr_ti_cox_model_path, hg_cox_model_path, egfr_components_cox_model_path
from pkgs.data.model_data_store import get_train_test_data
from pkgs.data.types import ExperimentScenario
from pkgs.experiments.utils import round_metric

def compute_time_dependent_auc(model: CoxTimeVaryingFitter | CoxPHFitter, data_train, data_test, duration_col, event_col, times):
    y_train = Surv.from_dataframe(event=event_col, time=duration_col, data=data_train)
    y_test = Surv.from_dataframe(event=event_col, time=duration_col, data=data_test)
    risk_scores_test = model.predict_partial_hazard(data_test).values.flatten()

    print(f"Risk scores test: {risk_scores_test.shape}")
    auc_values, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores_test, times)
    return auc_values, mean_auc

def run_cox_model(scenario: ExperimentScenario):
    assert scenario in [ExperimentScenario.TIME_VARIANT, ExperimentScenario.HETEROGENEOUS, ExperimentScenario.EGFR_COMPONENTS]

    data_train, data_test = get_train_test_data(scenario)

    model_path = get_model_path(scenario)

    if not os.path.exists(model_path):
        model = CoxTimeVaryingFitter(penalizer=0.1)

        print(f'Fitting model:\n')
        model.fit(data_train, event_col='has_esrd', id_col='subject_id')

        joblib.dump(model, model_path)
    else:
        model = joblib.load(model_path)

    print('Evaluate on test data')

    risk_scores_test = model.predict_partial_hazard(data_test)
    c_index_test = round_metric(concordance_index(data_test['duration_in_days'], -risk_scores_test, data_test['has_esrd']))
    print(f'Concordance Index Test: {c_index_test}')

    times = np.arange(1, 365, 1)

    _, mean_auc = compute_time_dependent_auc(model, data_train, data_test, 'duration_in_days', 'has_esrd', times)
    print(f"Mean time-dependent AUC: {mean_auc:.4f}")

def get_model_path(scenario: ExperimentScenario):
    assert scenario in [ExperimentScenario.TIME_VARIANT, ExperimentScenario.HETEROGENEOUS, ExperimentScenario.EGFR_COMPONENTS]

    model_path = {
        ExperimentScenario.TIME_VARIANT: egfr_tv_cox_model_path,
        ExperimentScenario.HETEROGENEOUS: hg_cox_model_path,
        ExperimentScenario.EGFR_COMPONENTS: egfr_components_cox_model_path
    }

    return model_path[scenario]

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

    times = np.arange(1, 365, 1)
    _, mean_auc = compute_time_dependent_auc(model, data_train, data_test, 'duration_in_days', 'has_esrd', times)

    print(f"Mean time-dependent AUC: {mean_auc:.4f}")

if __name__ == "__main__":
    print("\nRunning time-invariant Cox model evaluation with time-dependent AUC...")
    run_ti_cox_model()

    print("\nRunning time-variant Cox model evaluation with time-dependent AUC...")
    run_cox_model(ExperimentScenario.EGFR_COMPONENTS)
