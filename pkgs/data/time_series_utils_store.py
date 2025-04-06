from pkgs.commons import patients_file_path
import pandas as pd
import numpy as np

def get_dead_status(row, patients):
    pid = row['subject_id']
    
    # Check if the patient exists in the patients DataFrame
    patient_dod = patients.loc[patients['subject_id'] == pid, 'dod']
    
    if patient_dod.empty:
        # Raise an exception if the patient ID doesn't exist in the patients table
        raise ValueError(f"Patient with subject_id {pid} not found in patients table")
    
    # If dod is missing, consider the patient alive
    if pd.isna(patient_dod.values[0]):
        return 0  # Alive
    
    # If dod is present, compare time with dod
    return int(row['time'] >= pd.to_datetime(patient_dod.values[0]))

def calculate_duration_in_days(df):
    # `duration_in_days` is the time from the first lab record to the current lab record.
    df['duration_in_days'] = (df['time'] - df.groupby('subject_id')['time'].transform('min')).dt.total_seconds() / (60 * 60 * 24)

    return df

def sample_raw_df(df):
    unique_subjects = df['subject_id'].unique()

    n_sample = min(len(unique_subjects), 100)
    sampled_subjects = np.random.choice(unique_subjects, size=n_sample, replace=False)

    sampled_df = df[df['subject_id'].isin(sampled_subjects)]

    return sampled_df

if __name__ == '__main__':
    patients = pd.read_csv(patients_file_path)

