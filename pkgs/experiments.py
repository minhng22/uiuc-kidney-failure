import os

import joblib
import pandas as pd
from lifelines import CoxTimeVaryingFitter
from sksurv.ensemble import RandomSurvivalForest
from lifelines.utils import concordance_index

from pkgs.commons import lab_events_file_path, lab_codes_albumin, \
    chart_events_file_path, cox_model_path

from pkgs.data import get_train_test_data
import numpy as np

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


def get_y(df):
    print(df[df.isnull().any(axis=1)])
    arr = df.to_numpy()
    aux = [(e1,e2) for e1,e2 in arr]

    return np.array(aux, dtype=[('Has_ESRD', '?'), ('Survival_in_days', '<f8')])


def run_survival_rf():
    df, _ = get_train_test_data()
    df['has_esrd'] = df['has_esrd'].astype(bool)

    # Prepare the feature matrix (X) and the target vector (y)
    X = df[['duration_in_days', 'egfr']]
    y = get_y(df[['has_esrd', 'duration_in_days']])

    # Initialize and fit the Random Survival Forest model
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, max_depth=10)
    rsf.fit(X, y)

    # Evaluate the model
    # Note: concordance_index is a common metric for survival models
    c_index = concordance_index(df['duration_in_days'], -rsf.predict(X), df['has_esrd'])

    print(f'Concordance Index: {c_index}')

    # Display model performance
    print(f'Feature Importances: {rsf.feature_importances_}')

if __name__ == '__main__':
    run_survival_rf()
