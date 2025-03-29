import numpy as np
import os
from sklearn.model_selection import train_test_split
from pkgs.commons import (
    egfr_tv_train_data_path, egfr_tv_test_data_path, egfr_ti_train_data_path, egfr_ti_test_data_path,
    egfr_components_test_data_path, egfr_components_train_data_path,
    heterogen_train_data_path, heterogen_test_data_path,
)
from pkgs.data.types import ExperimentScenario
from pkgs.data.time_series_store import get_time_series_data_ckd_patients
import pandas as pd


# Pick a small subset of the data to test the models
# Random pick censored and uncensored patients.
def sample(df):
    num_subjects = 500

    esrd_patients = df[df['has_esrd'] == True]['subject_id'].unique()
    non_esrd_patients = df[~df['subject_id'].isin(esrd_patients)]['subject_id'].unique()
    print(
        f"Number of subjects with esrd: {len(esrd_patients)}\n"
        f"Number of subjects without esrd: {len(non_esrd_patients)}\n" 
        f"Total: {df['subject_id'].nunique()}")

    rand_subjects_esrd = np.random.choice(
        esrd_patients, size=num_subjects, replace=False)
    rand_subjects_no_esrd = np.random.choice(
        non_esrd_patients, size=num_subjects, replace=False)
    
    res = df[df['subject_id'].isin(np.concatenate((rand_subjects_esrd, rand_subjects_no_esrd), axis=0))]

    return res

def get_train_test_data(scenario: ExperimentScenario):
    train_data_stored_path = {
        ExperimentScenario.TIME_INVARIANT: egfr_ti_train_data_path,
        ExperimentScenario.TIME_VARIANT: egfr_tv_train_data_path,
        ExperimentScenario.HETEROGENEOUS: heterogen_train_data_path,
        ExperimentScenario.EGFR_COMPONENTS: egfr_components_train_data_path
    }
    test_data_stored_path = {
        ExperimentScenario.TIME_INVARIANT: egfr_ti_test_data_path,
        ExperimentScenario.TIME_VARIANT: egfr_tv_test_data_path,
        ExperimentScenario.HETEROGENEOUS: heterogen_test_data_path,
        ExperimentScenario.EGFR_COMPONENTS: egfr_components_test_data_path
    }
    train_path = train_data_stored_path[scenario]
    test_path = test_data_stored_path[scenario]

    print(f'Train data path {train_path}\nTest data path {test_path}')

    if not os.path.exists(train_path):
        data = get_time_series_data_ckd_patients(scenario)

        train_subjects, test_subjects = train_test_split(data['subject_id'].unique(), test_size=0.2, random_state=42)

        data_test = data[data['subject_id'].isin(test_subjects)]
        data_train = data[data['subject_id'].isin(train_subjects)]

        data_train.reset_index(drop=True, inplace=True)
        data_test.reset_index(drop=True, inplace=True)

        data_train.to_csv(train_path)
        data_test.to_csv(test_path)
    else:
        data_train = pd.read_csv(train_path)
        data_test = pd.read_csv(test_path)

    print(
        f'Number of patients: '
        f'test {data_test["subject_id"].nunique()} and train {data_train["subject_id"].nunique()}\n'
        f'Number of records: test {len(data_test)} and train {len(data_train)}'
    )

    return data_train, data_test

if __name__ == '__main__':
    get_train_test_data(ExperimentScenario.EGFR_COMPONENTS)