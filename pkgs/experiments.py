import os

import joblib
import pandas as pd
from lifelines import CoxTimeVaryingFitter

from pkgs.commons import lab_events_file_path, lab_codes_albumin, \
    chart_events_file_path, cox_model_path

from pkgs.data import get_train_test_data


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
    data_train, data_test = get_train_test_data()

    if not os.path.exists(cox_model_path):
        # Initialize the CoxPHFitter
        model = CoxTimeVaryingFitter()

        # Fit the model
        print(f'Fitting model:\n')
        model.fit(data_train, event_col='dead')

        joblib.dump(model, cox_model_path)
    else:
        model = joblib.load(cox_model_path)

    print(f'Number of test records: {len(data_test)}')

    # Calculate partial hazard for new data
    partial_hazard = model.predict_partial_hazard(data_test)

    # Aggregate partial hazards by individual
    data_test['partial_hazard'] = partial_hazard.values
    aggregated_hazard = data_test.groupby('subject_id')['partial_hazard'].sum()

    # To calculate the survival function, combine the aggregated hazard with the baseline survival function
    baseline_survival = model.baseline_survival_
    print(f'Baseline survival: \n{baseline_survival}')

    # Example time points to evaluate the survival function
    time_points = [5, 10, 15, 20, 25]

    # Calculate the survival function for each individual in data_test
    survival_functions = []

    for idx, (ind, hazard) in enumerate(aggregated_hazard.items()):
        survival_function = baseline_survival ** hazard
        survival_functions.append((ind, survival_function.loc[time_points]))

    # Print the survival function for each individual at the specified time points
    for ind, sf in survival_functions:
        print(f"Survival function for individual {ind}:")
        print(sf)


if __name__ == '__main__':
    run_cox_model()
