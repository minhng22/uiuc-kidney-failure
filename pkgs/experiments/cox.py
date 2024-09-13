from lifelines.utils import concordance_index
import joblib
from lifelines import CoxTimeVaryingFitter, CoxPHFitter
import os

from pkgs.commons import tv_cox_model_path
from pkgs.data.model_data_store import get_train_test_data
from pkgs.experiments.utils import report_metric

def prep_data(df):
    df['start'] = df.groupby('subject_id').cumcount() * df['duration_in_days']  # Calculate start times
    df['stop'] = df['start'] + df['duration_in_days']  # Calculate stop times
    df.dropna(inplace=True)
    # Adjust rows where start == stop by adding a small value to stop
    df.loc[df['start'] == df['stop'], 'stop'] += 1e-5  # Adding a small value to ensure stop > start
    return df

def run_time_varying_cox_model():
    data_train, data_test = get_train_test_data()
    data_train = prep_data(data_train)[['subject_id', 'start', 'stop', 'has_esrd', 'egfr']]
    data_test = prep_data(data_test)[['subject_id', 'start', 'stop', 'has_esrd', 'egfr']]

    if not os.path.exists(tv_cox_model_path):
        model = CoxTimeVaryingFitter()

        print(f'Fitting model:\n')
        model.fit(data_train, event_col='has_esrd', id_col='subject_id')

        joblib.dump(model, tv_cox_model_path)
    else:
        model = joblib.load(tv_cox_model_path)

    print('Evaluate on training data')
    risk_scores = model.predict_partial_hazard(data_train)
    c_index = report_metric(concordance_index(data_train['stop'], -risk_scores, data_train['has_esrd']))
    print(f'Concordance Index: {c_index}')

    print('Evaluate on test data')
    risk_scores_test = model.predict_partial_hazard(data_test)
    c_index_test = report_metric(concordance_index(data_test['stop'], -risk_scores_test, data_test['has_esrd']))
    print(f'Concordance Index Test: {c_index_test}')


def run_cox_model():
    data_train, data_test = get_train_test_data()
    data_train = prep_data(data_train)[['subject_id', 'start', 'stop', 'has_esrd', 'egfr']]
    data_test = prep_data(data_test)[['subject_id', 'start', 'stop', 'has_esrd', 'egfr']]

    if not os.path.exists(tv_cox_model_path):
        model = CoxPHFitter()

        print(f'Fitting model:\n')
        model.fit(data_train, event_col='has_esrd', id_col='subject_id')

        joblib.dump(model, tv_cox_model_path)
    else:
        model = joblib.load(tv_cox_model_path)

    print('Evaluate on training data')
    risk_scores = model.predict_partial_hazard(data_train)
    c_index = report_metric(concordance_index(data_train['stop'], -risk_scores, data_train['has_esrd']))
    print(f'Concordance Index: {c_index}')

    print('Evaluate on test data')
    risk_scores_test = model.predict_partial_hazard(data_test)
    c_index_test = report_metric(concordance_index(data_test['stop'], -risk_scores_test, data_test['has_esrd']))
    print(f'Concordance Index Test: {c_index_test}')


if __name__ == '__main__':
    run_time_varying_cox_model()

