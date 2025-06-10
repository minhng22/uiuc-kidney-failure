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
import matplotlib.pyplot as plt
import seaborn as sns
from pkgs.commons import esrd_patient_ids_path


# Pick a small subset of the data to test the models
# Random pick censored and uncensored patients.
def sample(df):
    num_subjects = 500

    esrd_ids_df = pd.read_csv(esrd_patient_ids_path)
    esrd_patient_ids = esrd_ids_df['subject_id'].unique()
    print(f"Number of subjects with esrd: {len(esrd_patient_ids)} {esrd_patient_ids[:5]}")

    non_esrd_patients = df[~df['subject_id'].isin(esrd_patient_ids)]['subject_id'].unique()
    print(
        f"Number of subjects with esrd: {len(esrd_patient_ids)}\n"
        f"Number of subjects without esrd: {len(non_esrd_patients)}\n" 
        f"Total: {df['subject_id'].nunique()}")

    rand_subjects_esrd = np.random.choice(
        esrd_patient_ids, size=num_subjects, replace=False)
    rand_subjects_no_esrd = np.random.choice(
        non_esrd_patients, size=num_subjects, replace=False)
    
    res = df[df['subject_id'].isin(np.concatenate((rand_subjects_esrd, rand_subjects_no_esrd), axis=0))]

    return res

def get_train_test_data(scenario: ExperimentScenario):
    train_data_stored_path = {
        ExperimentScenario.NON_TIME_VARIANT: egfr_ti_train_data_path,
        ExperimentScenario.TIME_VARIANT: egfr_tv_train_data_path,
        ExperimentScenario.HETEROGENEOUS: heterogen_train_data_path,
        ExperimentScenario.EGFR_COMPONENTS: egfr_components_train_data_path
    }
    test_data_stored_path = {
        ExperimentScenario.NON_TIME_VARIANT: egfr_ti_test_data_path,
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

    data_train.reset_index(drop=True, inplace=True)
    data_test.reset_index(drop=True, inplace=True)

    return data_train, data_test

def analyze_train_test_data():
    for scenario in [ExperimentScenario.NON_TIME_VARIANT, ExperimentScenario.TIME_VARIANT, ExperimentScenario.HETEROGENEOUS, ExperimentScenario.EGFR_COMPONENTS]:
        print(f"Analyzing scenario: {scenario}")
        data_train, data_test = get_train_test_data(scenario)
        print(f"Train data:\n{data_train.head()}")
        print(f"Test data:\n{data_test.head()}")

        print(f"Nan check in train data:\n{data_train.isna().sum()}")
        print(f"Nan check in test data:\n{data_test.isna().sum()}")

        # Analyze number of patients
        num_patients_train = data_train['subject_id'].nunique()
        num_patients_test = data_test['subject_id'].nunique()
        print(f"Number of patients in train data: {num_patients_train}")
        print(f"Number of patients in test data: {num_patients_test}")

        # Analyze number of records
        num_records_train = len(data_train)
        num_records_test = len(data_test)
        print(f"Number of records in train data: {num_records_train}")
        print(f"Number of records in test data: {num_records_test}")

        if scenario == ExperimentScenario.TIME_VARIANT:
            # Max duration in days
            max_duration_train = data_train['duration_in_days'].max()
            max_duration_test = data_test['duration_in_days'].max()
            print(f"Max duration in days in train data: {max_duration_train}")
            print(f"Max duration in days in test data: {max_duration_test}")

            # Analyze the distribution of eGFR values
            print(f"Distribution of eGFR in train data:\n{data_train['egfr'].describe()}")
            print(f"Distribution of eGFR in test data:\n{data_test['egfr'].describe()}")
            
            # Plot eGFR trajectories for random 1000 patients in test data
            sns.set(style="whitegrid")
            random_patients = np.random.choice(data_test['subject_id'].unique(), size=1000, replace=False)
            plt.figure(figsize=(18, 6))
            for patient in random_patients:
                patient_data_test = data_test[data_test['subject_id'] == patient]
                plt.plot(patient_data_test['duration_in_days'], patient_data_test['egfr'], label=f'Test {patient}', linestyle='--')
            
            plt.xlabel('Duration in days')
            plt.ylabel('eGFR')
            plt.title('eGFR Trajectories for Random 1000 Patients In Test Data')
            plt.legend()
            
            plt.savefig('generated_data/egfr_trajectories_test.png')
            plt.clf()

            # average eGFR over time of random_patients
            avg_egfr = data_test[data_test['subject_id'].isin(random_patients)].groupby('duration_in_days')['egfr'].mean()
            plt.xlabel('Duration in days')
            plt.ylabel('Average eGFR')
            plt.plot(avg_egfr.index, avg_egfr.values, label='Average eGFR', color='blue')
            plt.title('Average eGFR Over Time for Random 1000 Patients In Test Data')
            plt.legend()
            plt.savefig('generated_data/avg_egfr_trajectories_test.png')
            plt.clf()
        elif scenario == ExperimentScenario.EGFR_COMPONENTS:
            # Analyze the distribution of eGFR values
            print(f"Distribution of eGFR in train data:\n{data_train['serum_creatinine'].describe()}")
            print(f"Distribution of eGFR in test data:\n{data_test['serum_creatinine'].describe()}")

            # Analyze the distribution of age
            print(f"Distribution of age in train data:\n{data_train['age'].describe()}")
            print(f"Distribution of age in test data:\n{data_test['age'].describe()}")

            # Analyze distribution of gender
            print(f"Distribution of age in train data:\n{data_train['gender'].describe()}")
            print(f"Distribution of age in test data:\n{data_test['gender'].describe()}")
        elif scenario == ExperimentScenario.HETEROGENEOUS:
            # Analyze the distribution of eGFR values
            print(f"Distribution of eGFR in train data:\n{data_train[data_train['egfr_missing'] == 0]['egfr'].describe()}")
            print(f"Distribution of eGFR in test data:\n{data_test[data_test['egfr_missing'] == 0]['egfr'].describe()}")

            # Analyze the distribution of protein values
            print(f"Distribution of protein in train data:\n{data_train[data_train['protein_missing'] == 0]['protein'].describe()}")
            print(f"Distribution of protein in test data:\n{data_test[data_test['protein_missing'] == 0]['protein'].describe()}")

            # Analyze the distribution of albumin values
            print(f"Distribution of albumin in train data:\n{data_train[data_train['albumin_missing'] == 0]['albumin'].describe()}")
            print(f"Distribution of albumin in test data:\n{data_test[data_test['albumin_missing'] == 0]['albumin'].describe()}")
        elif scenario == ExperimentScenario.NON_TIME_VARIANT:
            # Analyze the distribution of eGFR values
            print(f"Distribution of eGFR in train data:\n{data_train['egfr'].describe()}")
            print(f"Distribution of eGFR in test data:\n{data_test['egfr'].describe()}")

def validate_subject_ids():
    # Make sure that the subject in train are not in test data
    for scenario in [ExperimentScenario.NON_TIME_VARIANT, ExperimentScenario.TIME_VARIANT, ExperimentScenario.HETEROGENEOUS, ExperimentScenario.EGFR_COMPONENTS]:
        print(f"Validating subject IDs for scenario: {scenario}")
        data_train, data_test = get_train_test_data(scenario)

        train_subjects = set(data_train['subject_id'].unique())
        test_subjects = set(data_test['subject_id'].unique())

        intersection = train_subjects.intersection(test_subjects)
        if intersection:
            print(f"Error: Found common subjects in train and test data for scenario {scenario}: {intersection}")
        else:
            print(f"No common subjects found in train and test data for scenario {scenario}.")

if __name__ == '__main__':
    analyze_train_test_data()