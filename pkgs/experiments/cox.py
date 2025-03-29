from lifelines.utils import concordance_index
import joblib
from lifelines import CoxTimeVaryingFitter, CoxPHFitter
import os

from pkgs.commons import egfr_tv_cox_model_path, egfr_ti_cox_model_path
from pkgs.data.model_data_store import get_train_test_data
from pkgs.experiments.utils import round_metric, evaluate_scikit_survival_model
from pkgs.data.types import ExperimentScenario

def get_callable_survival_functions(cox_model: CoxTimeVaryingFitter, data):
    pass

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
    evaluate_scikit_survival_model(
        data_test, model.predict_partial_hazard(data_test), get_callable_survival_functions(model, data_test), 
        data_train)


def run_ti_cox_model():
    data_train, data_test = get_train_test_data(ExperimentScenario.TIME_INVARIANT)

    if not os.path.exists(egfr_ti_cox_model_path):
        model = CoxPHFitter()

        print(f'Fitting model:\n')
        model.fit(data_train, duration_col= 'duration_in_days', event_col='has_esrd')

        joblib.dump(model, egfr_ti_cox_model_path)
    else:
        model = joblib.load(egfr_ti_cox_model_path)

    print('Evaluate on training data')
    risk_scores = model.predict_partial_hazard(data_train)
    c_index = round_metric(concordance_index(data_train['duration_in_days'], -risk_scores, data_train['has_esrd']))
    print(f'Concordance Index: {c_index}')

    print('Evaluate on test data')
    risk_scores_test = model.predict_partial_hazard(data_test)
    c_index_test = round_metric(concordance_index(data_test['duration_in_days'], -risk_scores_test, data_test['has_esrd']))
    print(f'Concordance Index Test: {c_index_test}')


if __name__ == '__main__':
    run_tv_cox_model()

