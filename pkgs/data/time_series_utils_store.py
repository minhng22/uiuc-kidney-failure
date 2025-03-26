from pkgs.commons import esrd_codes, patients_file_path
from pkgs.data.store import get_egfr_df, get_first_time_esrd_df, get_protein_df
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

# process patients who have not progressed to ESRD
def process_negative_patients(patient_ids, scenario_name):
    print(
        f"Processing patients who have not progressed to ESRD:\n"
        f"Number of patients: {len(patient_ids)}")

    patients = pd.read_csv(patients_file_path)
    patients = patients[patients['subject_id'].isin(patient_ids)]

    print(
        f"Stats on patients:\n"
        f"Total patients: {len(patients)}\n"
        f"Dead patients: {patients['dod'].isna().sum()}\n"
        f"Alive patients: {patients['dod'].notna().sum()}"
    )
    lab_df = get_lab_df_for_scenario_name(patients, scenario_name)

    lab_df['has_esrd'] = 0

    lab_df['time'] = pd.to_datetime(lab_df['time'])
    # `duration_in_days` is the time from the first lab record to the current lab record.
    lab_df['duration_in_days'] = (lab_df['time'] - lab_df.groupby('subject_id')['time'].transform('min')).dt.total_seconds() / (60 * 60 * 24)

    # drop subject where there are missing values in duration_in_days
    # In mimic-iv, these are:
    #            has_esrd  duration_in_days
    # 54828      False               NaN
    # 61737      False               NaN
    # 92289      False               NaN
    # 163716     False               NaN
    # 223727     False               NaN
    # 324347     False               NaN
    # 447693     False               NaN
    # These subject have not progressed to ESRD, and only has one record.
    lab_df = lab_df.groupby('subject_id').filter(lambda x: x['duration_in_days'].notna().all())
    print(
        f"Number of patients after filtering out NaN duration_in_days: {lab_df['subject_id'].nunique()}\n"
        f"Records sample:\n{lab_df[['subject_id', 'time', 'has_esrd']].head()}")

    print(
        f"Stats on eGFR:\n"
        f"Number of records: {len(lab_df)}. Number of patients: {lab_df['subject_id'].nunique()}\n"
        f"mean {lab_df['egfr'].mean():.3f} sd {lab_df['egfr'].std():.3f}")
    
    return lab_df[['subject_id', 'duration_in_days', 'egfr', 'has_esrd']]

def get_lab_df_for_scenario_name(patients, scenario_name):
    if scenario_name == 'time_variant':
        lab_df = get_egfr_df(patients)
    elif scenario_name == 'heterogeneous':
        egfr_df = get_egfr_df(patients)
        egfr_df['egfr_missing'] = 0
        egfr_df['protein_missing'] = 1
        egfr_df['protein'] = 0

        protein_df = get_protein_df(patients)
        protein_df['egfr_missing'] = 1
        protein_df['protein_missing'] = 0
        protein_df['egfr'] = 0
        
        lab_df = pd.concat([egfr_df, protein_df])
    else:
        assert scenario_name == 'egfr_components'
        lab_df = get_egfr_df(patients)
        lab_df['gender'] = lab_df['gender'].map({'M': 1, 'F': 0})
    
    lab_df.rename(columns={'anchor_age': 'age', 'charttime': 'time'}, inplace=True)

    print('Finished getting raw lab records for scenario:', scenario_name)
    print(lab_df.head())

    return lab_df
        
# process patients who have progressed to ESRD
def process_positive_patients(diagnoses_df, patient_ids, scenario_name):
    def validate(D):
        filtered_df = D[D['has_esrd'] == 1]
        id_patients_w_lab_records_esrd = filtered_df['subject_id'].unique()

        if len(id_patients_w_lab_records_esrd) != D['subject_id'].nunique():
            print(
                f"Difference"
            )
            for patient in list(set(D['subject_id'].unique()) - set(id_patients_w_lab_records_esrd)):
                print(
                    f"subject_id: {patient} is bad.\n"
                    f"data: {D[D['subject_id'] == patient][['subject_id', 'time', 'has_esrd', 'first_diagnose_esrd_time']]}\n")
    print(
        f"Processing patients who have progressed to ESRD:\n"
        f"Number of patients: {len(patient_ids)}")
    diagnose_esrd_df = diagnoses_df[diagnoses_df['icd_code'].isin(esrd_codes)]
    print(f"Number of patients with ESRD: {diagnose_esrd_df['subject_id'].nunique()}")
    first_time_esrd_df = get_first_time_esrd_df(diagnose_esrd_df)

    patients = pd.read_csv(patients_file_path)
    patients = patients[patients['subject_id'].isin(patient_ids)]

    print(
        f"Stats on patients:\n"
        f"Total patients: {len(patients)}\n"
        f"Dead patients: {patients['dod'].isna().sum()}\n"
        f"Alive patients: {patients['dod'].notna().sum()}"
    )

    lab_df = get_lab_df_for_scenario_name(patients, scenario_name)

    lab_df = pd.merge(lab_df, first_time_esrd_df, on='subject_id', how='left')
    print(
        f'Number of patients after merging: {lab_df["subject_id"].nunique()}\n'
    )

    lab_df['time'] = pd.to_datetime(lab_df['time'])
    lab_df['first_diagnose_esrd_time'] = pd.to_datetime(lab_df['first_diagnose_esrd_time'])

    # Note: This does not work
    # lab_df = lab_df[lab_df['time'] <= lab_df['first_diagnose_esrd_time']]
    # Because a lot of patients don't have lab event records for the admission in which they are diagnosed with ESRD.
    # Example:
    # Patient 11206658: 
    #         subject_id  labevent_id     hadm_id                 time first_diagnose_esrd_time
    # 188809    11206658   14083695.0  21087283.0  2200-02-11 18:20:00      2202-10-11 19:07:00
    # 188810    11206658   14083771.0  21087283.0  2200-02-12 08:07:00      2202-10-11 19:07:00
    # 188811    11206658   14083785.0  21087283.0  2200-02-14 07:45:00      2202-10-11 19:07:00
    # 188812    11206658   14083799.0  21087283.0  2200-02-16 07:00:00      2202-10-11 19:07:00
    # 188813    11206658   14083836.0  21087283.0  2200-02-17 07:55:00      2202-10-11 19:07:00
    # 188814    11206658   14083873.0         NaN  2202-10-11 16:40:00      2202-10-11 19:07:00

    lab_df['has_esrd'] = lab_df['time'] >= lab_df['first_diagnose_esrd_time']
    lab_df['has_esrd'] = lab_df['has_esrd'].astype(int)

    # `duration_in_days` is the time from the first lab record to the current lab record.
    lab_df['duration_in_days'] = (lab_df['time'] - lab_df.groupby('subject_id')['time'].transform('min')).dt.total_seconds() / (60 * 60 * 24)

    # empty value means they only have one record.
    lab_df = lab_df.groupby('subject_id').filter(lambda x: x['duration_in_days'].notna().all())

    # These are patients who we have lab records prior to their first diagnose of ESRD.
    progressed_patients_ids = lab_df.loc[lab_df['has_esrd'] == 1, 'subject_id'].unique()
    lab_df = lab_df[lab_df['subject_id'].isin(progressed_patients_ids)]
    print(f"Number of ESRD who have lab records at or prior to their esrd diagnose: {len(progressed_patients_ids)}")

    # Filter out records after the first 'has_esrd' == 1
    lab_df = lab_df.sort_values(by=['subject_id', 'time'])
    def filter_records(g):
        first_esrd_index = g[g['has_esrd'] == 1].index.min()
        if pd.isna(first_esrd_index):
            print(f"subject_id: {g['subject_id'].iloc[0]} has no 'has_esrd' == 1")
            return g
        return g.loc[g.index <= first_esrd_index]
    lab_df = lab_df.groupby('subject_id', group_keys=False).apply(filter_records)
    print(
        f"Number of patients after filtering out records after the first 'has_esrd' == 1: {lab_df['subject_id'].nunique()}\n"
        f"Number of records: {len(lab_df)}. Records sample:\n{lab_df[['subject_id', 'time', 'has_esrd', 'first_diagnose_esrd_time']].head()}")

    validate(lab_df)

    print(
        f"Stats on eGFR:\n"
        f"Number of records: {len(lab_df)}. Number of patients: {lab_df['subject_id'].nunique()}\n"
        f"mean {lab_df['egfr'].mean():.3f} sd {lab_df['egfr'].std():.3f}")

    return lab_df[['subject_id', 'duration_in_days', 'egfr', 'has_esrd']]

def sample_raw_df(df):
    unique_subjects = df['subject_id'].unique()

    n_sample = min(len(unique_subjects), 100)
    sampled_subjects = np.random.choice(unique_subjects, size=n_sample, replace=False)

    sampled_df = df[df['subject_id'].isin(sampled_subjects)]

    return sampled_df

if __name__ == '__main__':
    patients = pd.read_csv(patients_file_path)
    get_lab_df_for_scenario_name(patients, "heterogeneous")
