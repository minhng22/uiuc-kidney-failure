import os

import joblib
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.utils import to_long_format

from pkgs.commons import lab_events_file_path, lab_codes_albumin, \
    chart_events_file_path, train_data_path, test_data_path, cox_model_path
from pkgs.data import get_time_series_data_esrd_patients
from sklearn.metrics import root_mean_squared_error

# No records of albumin for patients progressed from ckd 3-5 to esrd.
def verify_that_albumin_records_not_exist_for_patients(patient_ids):
    lab_events_df = pd.read_csv(lab_events_file_path)
    lab_albumin_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_albumin)]
    print(f'Number of records for albumin: {len(lab_albumin_df)}')

    chart_events_df = pd.read_csv(chart_events_file_path)
    chart_events_df = chart_events_df[chart_events_df['subject_id'].isin(patient_ids)]

    chart_albumin_df = chart_events_df[chart_events_df['itemid'].isin(lab_codes_albumin)]
    print(f'Number of records for albumin: {len(chart_albumin_df)}')


def run_cox_model():
    if not os.path.exists(train_data_path):
        data = get_time_series_data_esrd_patients()

        # Find dead patients
        dead_patient_ids = data.groupby('subject_id').filter(lambda x: (x['dead'] == 1).all())[
            'subject_id'].unique().tolist()
        alive_patient_ids = data[~data['subject_id'].isin(dead_patient_ids)]['subject_id'].unique().tolist()
        test_data_ids = alive_patient_ids[: len(alive_patient_ids) // 5]

        data_test = data[data['subject_id'].isin(test_data_ids)]  # 80/20 split
        data_train = data[~data['subject_id'].isin(data_test['subject_id'].unique())]

        print(
            f'Number of test {len(data_test['subject_id'].unique())} and train {len(data_train['subject_id'].unique())}\n'
            f'Number of alive patients: {len(alive_patient_ids)} and test patients: {len(test_data_ids)}\n'
            f'Number of test patients records {len(data_test)}'
        )

        data_train.to_csv(train_data_path)
        data_test.to_csv(test_data_path)

        data_test = data_test.sort_values('time').groupby('subject_id').last().reset_index()
        print(f'Number of test patient records after group: {len(data_test)}')
    else:
        data_train = pd.read_csv(train_data_path)
        data_test = pd.read_csv(test_data_path)

    data_train = to_long_format(data_train, duration_col='age')[['subject_id', 'egfr', 'dead', 'start', 'stop']].copy()
    data_test = to_long_format(data_test, duration_col='age')[['subject_id', 'egfr', 'dead', 'start', 'stop']].copy()

    print(f'data_train in long format: \n{data_train.head()}')
    print(f'data_test in long format: \n{data_test.head()}')

    if not os.path.exists(cox_model_path):
        # Initialize the CoxPHFitter
        cph = CoxTimeVaryingFitter()

        # Fit the model
        print(f'Fitting model:\n')
        cph.fit(data_train, event_col='dead')

        # Print the summary
        cph.print_summary()

        joblib.dump(cph, cox_model_path)
    else:
        cph = joblib.load(cox_model_path)

    print(f'Number of test records: {len(data_test)}')

    M = cph.predict_partial_hazard(data_test)
    print(f"len of predict_median: \n{len(M)}")


if __name__ == '__main__':
    run_cox_model()
