import os

import joblib
import pandas as pd
from lifelines import CoxTimeVaryingFitter
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from lifelines.utils import concordance_index

from pkgs.commons import lab_events_file_path, lab_codes_albumin, \
    chart_events_file_path, cox_model_path, srf_model_path, gbsa_model_path

from pkgs.data.model_data_supply import get_train_test_data, mini
import numpy as np
import datetime


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
    return np.array(list(zip(df['has_esrd'].astype(bool), df['duration_in_days'])), 
              dtype=[('event', bool), ('time', np.float64)])


# Function to check if any NaN exists in the 'time' field
def contains_nan_in_field(array, field_name):
    return np.isnan(array[field_name]).any()


# Data needs to not be time-invariant setup
def run_survival_rf():
    df, df_test = get_train_test_data()
    df['has_esrd'] = df['has_esrd'].astype(bool)
    df = mini(df)
    X = df[['duration_in_days', 'egfr']]
    y = get_y(df)

    if os.path.exists(srf_model_path):
        rsf = joblib.load(srf_model_path)
    else:
        print(f'Fitting Random Survival Forest model. Current time {datetime.datetime.now()}:\n')
        rsf = RandomSurvivalForest(n_jobs= 10, verbose=2)
        rsf.fit(X, y)
        joblib.dump(rsf, srf_model_path)

    # C-index is the most popular metric in the last 14 years by a wide margin for evaluating survival models.
    # ref: https://journal.r-project.org/articles/RJ-2023-009/RJ-2023-009.pdf
    c_index = concordance_index(df['duration_in_days'], -rsf.predict(X), df['has_esrd'])
    print(f'Concordance Index: {c_index}')

    df_test['has_esrd'] = df_test['has_esrd'].astype(bool)
    X_test = df_test[['duration_in_days', 'egfr']]

    c_index_test = concordance_index(df_test['duration_in_days'], -rsf.predict(X_test), df_test['has_esrd'])
    print(f'Concordance Index Test: {c_index_test}')

    eval_duration(df_test)
    eval_duration(df)

    surv_fns = rsf.predict_survival_function(X_test)
    # Initialize a list to store median survival times
    median_survival_times = []

    # Loop through each survival function
    for surv_fn in surv_fns:
        # Convert the survival function to an array of times and probabilities
        times = np.array([time for time in surv_fn.x])
        survival_probs = np.array([prob for prob in surv_fn.y])
        
        # Find the time where the survival probability drops to 0.5 or below
        median_time = times[survival_probs <= 0.5][0] if any(survival_probs <= 0.5) else np.nan
        
        # Append the median survival time to the list
        median_survival_times.append(median_time)

    # Convert to numpy array or other format if needed
    median_survival_times = np.array(median_survival_times)

    # Now you have the median survival times for each subject
    print(median_survival_times.shape)


def run_gbsa():
    df, df_test = get_train_test_data()
    df = mini(df)
    X = df[['duration_in_days', 'egfr']].to_numpy()
    y = get_y(df)

    if os.path.exists(gbsa_model_path):
        gbsa = joblib.load(gbsa_model_path)
    else:
        print('Fitting Gradient Boosting Survival Analysis model')
        gbsa = GradientBoostingSurvivalAnalysis()
        gbsa.fit(X, y)
        joblib.dump(gbsa, gbsa_model_path)

    c_index = concordance_index(df['duration_in_days'], -gbsa.predict(X), df['has_esrd'])
    print(f'Concordance Index: {c_index}')

    df_test['has_esrd'] = df_test['has_esrd'].astype(bool)
    df_test.dropna(inplace=True)
    X_test = df_test[['duration_in_days', 'egfr']].to_numpy()
    print(np.isnan(X_test).any())

    c_index_test = concordance_index(df_test['duration_in_days'], -gbsa.predict(X_test), df_test['has_esrd'])
    print(f'Concordance Index Test: {c_index_test}')    


def eval_duration(df):
    esrd_subjects_df = df[df['has_esrd'] == True]
    max_duration = esrd_subjects_df['duration_in_days'].max()
    print(f"max duration: {max_duration}")

if __name__ == '__main__':
    run_gbsa()
