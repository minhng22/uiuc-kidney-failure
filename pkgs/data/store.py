import pandas as pd
from pkgs.commons import ckd_codes, ckd_codes_stage3_to_5, diagnose_icd_file_path, esrd_codes, lab_events_file_path, lab_codes_creatinine, admissions_file_path, patients_file_path
from pkgs.data.utils_store import filter_df_on_icd_code
from pkgs.data.utils import calculate_eGFR
import numpy as np


# @ethnicity_to_race - if True:
# (1) filters out patients with selection 'PATIENT DECLINED TO ANSWER', 'UNABLE TO OBTAIN', 'UNKNOWN'
# (2) information in admission.csv is actually ethnicity information.
#       Convert it to race: 'ASIAN - ASIAN INDIAN' -> 'ASIAN'
def get_admission_df(ethnicity_to_race: bool):
    admission_df = pd.read_csv(admissions_file_path)

    bad_record_admission_df = admission_df[
        admission_df['race'].isin(["PATIENT DECLINED TO ANSWER", "UNABLE TO OBTAIN", "UNKNOWN"])]
    percentage_filtered = (len(bad_record_admission_df) / len(admission_df)) * 100

    #print(f"percentage of patients with race selection 'PATIENT DECLINED TO ANSWER', "f"'UNABLE TO OBTAIN', or 'UNKNOWN': {percentage_filtered:.2f}%")
    admission_df = admission_df[
        ~admission_df['race'].isin(["PATIENT DECLINED TO ANSWER", "UNABLE TO OBTAIN", "UNKNOWN"])]

    if ethnicity_to_race:
        ethnicity_to_race = {
            "BLACK/AFRICAN AMERICAN": "BLACK",
            "BLACK/CAPE VERDEAN": "BLACK",
            "BLACK/CARIBBEAN ISLAND": "BLACK",
            "BLACK/AFRICAN": "BLACK",
            "WHITE - RUSSIAN": "WHITE",
            "WHITE - OTHER EUROPEAN": "WHITE",
            "WHITE - EASTERN EUROPEAN": "WHITE",
            "WHITE - BRAZILIAN": "WHITE",
            "HISPANIC/LATINO - PUERTO RICAN": "HISPANIC/LATINO",
            "HISPANIC OR LATINO": "HISPANIC/LATINO",
            "HISPANIC/LATINO - DOMINICAN": "HISPANIC/LATINO",
            "HISPANIC/LATINO - GUATEMALAN": "HISPANIC/LATINO",
            "HISPANIC/LATINO - SALVADORAN": "HISPANIC/LATINO",
            "HISPANIC/LATINO - HONDURAN": "HISPANIC/LATINO",
            "HISPANIC/LATINO - CUBAN": "HISPANIC/LATINO",
            "HISPANIC/LATINO - CENTRAL AMERICAN": "HISPANIC/LATINO",
            "HISPANIC/LATINO - COLUMBIAN": "HISPANIC/LATINO",
            "HISPANIC/LATINO - MEXICAN": "HISPANIC/LATINO",
            "ASIAN - CHINESE": "ASIAN",
            "ASIAN - SOUTH EAST ASIAN": "ASIAN",
            "ASIAN - ASIAN INDIAN": "ASIAN",
            "ASIAN - KOREAN": "ASIAN"
        }

        admission_df['race'] = admission_df['race'].replace(ethnicity_to_race)

    return admission_df


def get_lab_events_df_for_patients(patient_df):
    lab_events_df = pd.read_csv(lab_events_file_path)
    lab_events_df = lab_events_df[lab_events_df['subject_id'].isin(patient_df['subject_id'])]
    lab_events_df['itemid'] = lab_events_df['itemid'].astype(str)
    lab_events_df['valuenum'] = lab_events_df['valuenum'].astype(float)

    return lab_events_df


def get_egfr_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)

    egfr_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_creatinine)]
    egfr_df = pd.merge(egfr_df, patient_df, on='subject_id', how='outer')
    egfr_df = egfr_df[egfr_df['valuenum'] != 0]
    egfr_df['egfr'] = egfr_df.apply(calculate_eGFR, axis=1)

    egfr_df.dropna(inplace=True)
    egfr_df = egfr_df.replace('', np.nan).dropna()

    return egfr_df

def get_protein_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)

    egfr_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_creatinine)]
    egfr_df = pd.merge(egfr_df, patient_df, on='subject_id', how='outer')
    egfr_df = egfr_df[egfr_df['valuenum'] != 0]
    egfr_df['egfr'] = egfr_df.apply(calculate_eGFR, axis=1)

    egfr_df.dropna(inplace=True)
    egfr_df = egfr_df.replace('', np.nan).dropna()

    return egfr_df

def get_first_time_esrd_df(diagnose_df):
    admission_df = pd.read_csv(admissions_file_path)
    admission_df['admittime'] = pd.to_datetime(admission_df['admittime'])
    # Initialize an empty list to store the results
    results = []

    # Loop through each patient in lab_df
    for subject_id, group in diagnose_df.groupby('subject_id'):
        match_rows = group[group['icd_code'].isin(esrd_codes)].iloc
        first_time_esrd = None

        for row in match_rows:
            hadm_id = row['hadm_id']
            admit_time = admission_df.loc[admission_df['hadm_id'] == hadm_id, 'admittime'].values[0]

            if first_time_esrd is None or admit_time < first_time_esrd:
                first_time_esrd = admit_time

        results.append({'subject_id': subject_id, 'first_diagnose_esrd_time': first_time_esrd})

    # Convert the results to a DataFrame if needed
    results_df = pd.DataFrame(results)
    print(
        f"first time having ESRD df:\n {results_df.head()}\n"
        f"Number of patients: {results_df['subject_id'].nunique()}")

    results_df.dropna()
    print(
        f"Number of patients after drop n/a: {results_df['subject_id'].nunique()}")

    return results_df


def get_ckd_patients_and_diagnoses(late_stage: bool = True):
    diagnoses_df = pd.read_csv(diagnose_icd_file_path)

    ckd_filter_codes = ckd_codes_stage3_to_5 if late_stage else ckd_codes

    ckd_diagnose_df = diagnoses_df[diagnoses_df['icd_code'].isin(ckd_filter_codes)]
    print(
        f"number of CKD subjects: {ckd_diagnose_df['subject_id'].nunique()}\n"
        f"percentage of subjects in dataset: {ckd_diagnose_df['subject_id'].nunique() / diagnoses_df['subject_id'].nunique() * 100:.3f}"
    )

    patients_df = pd.read_csv(patients_file_path)
    patients_df = patients_df[patients_df['subject_id'].isin(ckd_diagnose_df['subject_id'].unique())]

    print(f"number of subjects (for validation): {patients_df['subject_id'].nunique()}")

    return patients_df, ckd_diagnose_df


def get_esrd_patients_and_diagnoses():
    diagnoses_df = pd.read_csv(diagnose_icd_file_path)

    esrd_diagnose_df = filter_df_on_icd_code(diagnoses_df, esrd_codes, ckd_codes)
    esrd_diagnose_df = esrd_diagnose_df[esrd_diagnose_df['icd_code'].isin(esrd_codes)]
    print(
        f"number of ESRD subjects: {esrd_diagnose_df['subject_id'].nunique()}\n"
        f"percentage of subjects in dataset: {esrd_diagnose_df['subject_id'].nunique() / diagnoses_df['subject_id'].nunique() * 100:.3f}"
    )

    patients_df = pd.read_csv(patients_file_path)
    patients_df = patients_df[patients_df['subject_id'].isin(esrd_diagnose_df['subject_id'].unique())]

    print(f"number of subjects (for validation): {patients_df['subject_id'].nunique()}")

    return patients_df, esrd_diagnose_df