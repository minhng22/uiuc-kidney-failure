import numpy as np
import os
from sklearn.model_selection import train_test_split
from pkgs.commons import (
    tv_train_data_path, tv_test_data_path, ti_train_data_path, ti_test_data_path
)
from pkgs.data.time_series_store import get_time_series_data_ckd_patients
import pandas as pd


# Pick a small subset of the data to test the models
# Random pick censored and uncensored patients.
def mini(df):
    num_subjects = 50

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


def get_tv_train_test_data():
    if not os.path.exists(tv_train_data_path):
        data = get_time_series_data_ckd_patients()

        train_subjects, test_subjects = train_test_split(data['subject_id'].unique(), test_size=0.2, random_state=42)

        data_test = data[data['subject_id'].isin(test_subjects)]
        data_train = data[data['subject_id'].isin(train_subjects)]

        data_train.reset_index(drop=True, inplace=True)
        data_test.reset_index(drop=True, inplace=True)

        data_train.to_csv(tv_train_data_path)
        data_test.to_csv(tv_test_data_path)
    else:
        data_train = pd.read_csv(tv_train_data_path)
        data_test = pd.read_csv(tv_test_data_path)

    print(
        f'Number of patients: '
        f'test {data_test["subject_id"].nunique()} and train {data_train["subject_id"].nunique()}\n'
        f'Number of records: test {len(data_test)} and train {len(data_train)}'
    )

    def validate(D):
        # Group by 'subject_id' and filter groups with fewer than 2 rows
        subjects_less_than_2_rows = D.groupby('subject_id').filter(lambda x: len(x) < 2)

        # Get the unique subject IDs with fewer than 2 rows
        for patient in subjects_less_than_2_rows['subject_id'].unique()[:5]:
            print(
                f"subject_id: {patient} is bad.\n"
                f"data: {subjects_less_than_2_rows[subjects_less_than_2_rows['subject_id'] == patient][['subject_id', 'time', 'has_esrd']]}\n")
    
    validate(data_train)
    validate(data_test)

    return data_train, data_test

def get_ti_train_test_data():
    if not os.path.exists(ti_train_data_path):
        data = get_time_series_data_ckd_patients()

        train_subjects, test_subjects = train_test_split(data['subject_id'].unique(), test_size=0.2, random_state=42)

        data_test = data[data['subject_id'].isin(test_subjects)]
        data_train = data[data['subject_id'].isin(train_subjects)]

        data_test = data_test.loc[data_test.groupby('subject_id')['duration_in_days'].idxmax()]
        data_train = data_train.loc[data_train.groupby('subject_id')['duration_in_days'].idxmax()]

        data_train.reset_index(drop=True, inplace=True)
        data_test.reset_index(drop=True, inplace=True)

        data_train.to_csv(ti_train_data_path)
        data_test.to_csv(ti_test_data_path)
    else:
        data_train = pd.read_csv(ti_train_data_path)
        data_test = pd.read_csv(ti_test_data_path)

    print(
        f'Number of patients: '
        f'test {data_test["subject_id"].nunique()} and train {data_train["subject_id"].nunique()}\n'
        f'Number of records: test {len(data_test)} and train {len(data_train)}'
    )

    return data_train, data_test


if __name__ == '__main__':
    get_ti_train_test_data()