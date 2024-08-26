import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from pkgs.commons import (
    diagnose_icd_file_path, patients_file_path,
    age_bins, esrd_codes,
    ckd_codes, admissions_file_path, ckd_codes_stage3_to_5, ckd_codes_hypertension, ckd_codes_diabetes_mellitus,
    lab_events_file_path, lab_codes_creatinine, prescription_file_path, ace_inhibitor_drugs, figs_path_icd_stats,
    regressor_model_train_data_path, regressor_model_test_data_path,
)
from lifelines.utils import to_long_format

def get_train_test_data_regressor_model():
    if not os.path.exists(regressor_model_train_data_path):
        data = get_time_series_data_ckd_patients()

        test_data_ids = data['subject_id'].unique()[: data['subject_id'].nunique() // 5] # 80/20 split

        data_test = data[data['subject_id'].isin(test_data_ids)]
        data_train = data[~data['subject_id'].isin(data_test['subject_id'].unique())]

        print(
            f'Number of test {len(data_test['subject_id'].unique())} and train {len(data_train['subject_id'].unique())}\n'
            f'Number of test patients records {len(data_test)}'
        )

        data_train.to_csv(regressor_model_train_data_path)
        data_test.to_csv(regressor_model_test_data_path)

        data_test = data_test.sort_values('time').groupby('subject_id').last().reset_index()
        print(f'Number of test patient records after group: {len(data_test)}')
    else:
        data_train = pd.read_csv(regressor_model_train_data_path)
        data_test = pd.read_csv(regressor_model_test_data_path)

    data_train = to_long_format(data_train, duration_col='age')[['subject_id', 'egfr', 'start', 'stop']].copy()
    data_test = to_long_format(data_test, duration_col='age')[['subject_id', 'egfr', 'start', 'stop']].copy()

    print(f'data_train in long format: \n{data_train.head()}')
    print(f'data_test in long format: \n{data_test.head()}')

    return data_train, data_test

def get_first_time_esrd(lab_df):
    admission_df = pd.read_csv(admissions_file_path)
    # Initialize an empty list to store the results
    results = []

    # Loop through each patient in lab_df
    for subject_id, group in lab_df.groupby('subject_id'):
        # Find the first row where 'icd_code' is in ICD_CODES
        match_row = group[group['icd_code'].isin(esrd_codes)].iloc[0]

        # Get the 'hadm_id' of that row
        hadm_id = match_row['hadm_id']

        # Query admission_df to find the 'admittime' of that 'hadm_id'
        admit_time = admission_df.loc[admission_df['hadm_id'] == hadm_id, 'admittime'].values[0]

        # Append the result to the list
        results.append({'subject_id': subject_id, 'first_diagnose_esrd_time': admit_time})

    # Convert the results to a DataFrame if needed
    results_df = pd.DataFrame(results)
    print(f"first time having ESRD df:\n {results_df.head()}")

    return results_df



# get late stage ckd patients and info of their progression to esrd.
# only_esrd set to True returns only patients who have progressed to ESRD.
def get_time_series_data_ckd_patients(only_esrd: bool = True):
    diagnoses_df = pd.read_csv(diagnose_icd_file_path)
    diagnoses_df = diagnoses_df[diagnoses_df['icd_code'].isin(ckd_codes_stage3_to_5 + esrd_codes)]

    esrd_patients = diagnoses_df[diagnoses_df['icd_code'].isin(esrd_codes)]['subject_id'].unique()
    print(f"Sample patients with esrd: {esrd_patients[:20]}")
    print(f'Number of patients progressed from ckd stage 3-5 to esrd are {len(esrd_patients)} '
          f'over {diagnoses_df["subject_id"].nunique()}, '
          f'accounts for {round(100 * len(esrd_patients)/diagnoses_df["subject_id"].nunique(), 3)}%')

    first_time_esrd_df = None
    if only_esrd:
        # filter late stage patients who progressed to esrd.
        diagnoses_df = diagnoses_df[diagnoses_df['icd_code'].isin(esrd_codes)]
        first_time_esrd_df = get_first_time_esrd(diagnoses_df)

    patients = pd.read_csv(patients_file_path)
    patients = patients[patients['subject_id'].isin(diagnoses_df["subject_id"].unique())]
    print(f"Number of patients: {len(patients)}")
    patients = add_race_to_patients(patients)

    lab_df = get_egfr_df(patients)
    print(f"EGFR data layout:\n{lab_df.columns}")
    print(
        f"Stats on eGFR:\n"
        f"Number of records: {len(lab_df)}. Number of patients: {lab_df['subject_id'].nunique()}\n"
        f"mean {lab_df['egfr'].mean():.3f} sd {lab_df['egfr'].std():.3f}")

    lab_df.rename(columns={'anchor_age': 'age', 'charttime': 'time'}, inplace=True)

    if only_esrd:
        lab_df = pd.merge(lab_df, first_time_esrd_df, on='subject_id', how='left')
        lab_df['has_esrd'] = lab_df['time'] >= lab_df['first_diagnose_esrd_time']

    lab_df.drop(columns=[
        'labevent_id', 'hadm_id', 'specimen_id', 'itemid', 'order_provider_id', 'storetime',
        'value', 'valuenum', 'valueuom', 'ref_range_lower', 'ref_range_upper', 'flag', 'priority', 'comments',
        'gender', 'dod', 'race', 'anchor_year', 'anchor_year_group', 'age',
        'first_diagnose_esrd_time' # keep this column if needed
    ], inplace=True)
    print(f"EGFR data layout:\n{lab_df.columns}")

    print(
        f'Final data: \n{lab_df.head()}\n'
        f'Number of records: {len(lab_df)}\n'
        f'Number of patients: {lab_df['subject_id'].nunique()}\n'
    )

    return lab_df


def medication_use(patient_df):
    pres_df = pd.read_csv(prescription_file_path)
    pres_df = pres_df[pres_df['subject_id'].isin(patient_df['subject_id'])]
    pres_df = pres_df[pres_df['drug'].isin(ace_inhibitor_drugs)]
    print(
        f"Number of records for ACE inhibitors: {len(pres_df)}. "
        f"Number of patients: {pres_df['subject_id'].nunique()}. "
        f"Percentage of patients: {pres_df['subject_id'].nunique() / patient_df['subject_id'].nunique() * 100:.3f} percent"
    )
    pass

def analyze_esrd():
    patients_df, diagnoses_df = get_esrd_patients_and_diagnoses()

    #plot_icd_codes(diagnoses_df)
    # age_statistics(patients_df)
    # gender_statistics(patients_df)
    # ethnicity_and_race_statistics(patients_df, True, True)

    #clinical_characteristic_analysis_esrd(esrd=True, num_patient_in_cohort=diagnoses_df['subject_id'].nunique())
    laboratory_params(patients_df)
    #medication_use(patients_df)


# Function to extract the digit value from a string and return it as a float.
# This is needed where, for example, a eGFR value in omr.csv is stored as ">60"
def extract_num_from_value(value):
    # Find all numbers in the string
    numbers = re.findall(r'\d+\.?\d*', value)
    # Take the last number found if any
    if numbers:
        return float(numbers[-1])
    else:
        return float('nan')


# calculate eGFR using the CKD-EPI formula
def calculate_eGFR(row):
    race = row['race']
    gender = 'FEMALE' if row['gender'] == 'F' else 'MALE'
    age = row['anchor_age']
    serum_creatinine = row['valuenum']
    serum_creatinine_unit = row['valueuom']

    assert serum_creatinine != 0, f"bad value of serum_creatinine {row['subject_id']}"

    # CKD-EPI constants
    if gender == 'male':
        k = 0.9
        alpha = -0.411
        constant = 1.0
    else:
        k = 0.7
        alpha = -0.329
        constant = 1.018

    if race == 'black':
        constant *= 1.159

    # CKD-EPI equation
    eGFR = 141 * min(serum_creatinine / k, 1) ** alpha * max(serum_creatinine / k,
                                                             1) ** -1.209 * 0.993 ** age * constant

    return eGFR


def get_lab_events_for_patients(patient_df):
    lab_events_df = pd.read_csv(lab_events_file_path)
    lab_events_df = lab_events_df[lab_events_df['subject_id'].isin(patient_df['subject_id'])]
    lab_events_df['itemid'] = lab_events_df['itemid'].astype(str)
    lab_events_df['valuenum'] = lab_events_df['valuenum'].astype(float)

    return lab_events_df

def get_egfr_df(patient_df):
    lab_events_df = get_lab_events_for_patients(patient_df)

    egfr_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_creatinine)]
    egfr_df = pd.merge(egfr_df, patient_df, on='subject_id', how='outer')
    egfr_df = egfr_df[egfr_df['valuenum'] != 0]
    egfr_df['egfr'] = egfr_df.apply(calculate_eGFR, axis=1)

    egfr_df.dropna()

    return egfr_df


def add_race_to_patients(patient_df, verbose:bool=False):
    race_df = ethnicity_and_race_statistics(patient_df, True)
    patient_df = pd.merge(patient_df, race_df, on='subject_id', how='outer')
    if verbose:
        print(f'Patients df with race: \n{patient_df[['subject_id', 'race', 'gender', 'anchor_age']].head()}')

    return patient_df

def laboratory_params(patient_df):
    patient_df = add_race_to_patients(patient_df)
    lab_events_df = get_lab_events_for_patients(patient_df)

    # serum creatinine
    sc_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_creatinine)]
    print(f"Number of records for Serum Creatinine: {len(sc_df)}")
    print(f"units: {sc_df['valueuom'].value_counts()}")
    print(
        f"Stats on Serum Creatinine:\n"
        f"Number of records: {len(sc_df)}\n"
        f"mean {sc_df['valuenum'].mean():.3f} sd {sc_df['valuenum'].std():.3f}")

    # omr.csv only has ~ 90 records.
    # all records in labevents.csv has null values for eGFR.
    # Looking at the `comment` column in labevents.csv, we get extract egfr from 'serum creatinine', 'age' and 'sex'.
    egfr_df = get_egfr_df(patient_df)
    print(
        f"Stats on eGFR:\n"
        f"Number of records: {len(egfr_df)}\n"
        f"mean {egfr_df['egfr'].mean():.3f} sd {egfr_df['egfr'].std():.3f}")

    # # 24hr urine protein
    # protein_24hr_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_proteins_24hr)]
    # print(f"Number of records for 24hr urine protein: {len(protein_24hr_df)}")
    # protein_24hr_df['valuenum'] = protein_24hr_df['valuenum'] / 1000 # mg/24hr to g/24hr
    #
    # print(f"units: {protein_24hr_df['valueuom'].value_counts()}")
    # print(
    #     f"Stats on 24hr urine protein:\n"
    #     f"Number of records: {len(protein_24hr_df)}\n"
    #     f"median {protein_24hr_df['valuenum'].median():.3f} IQR {(protein_24hr_df['valuenum'].quantile(0.75) - protein_24hr_df['valuenum'].quantile(0.25)):.3f}")


def clinical_characteristic_analysis_esrd(esrd: bool, num_patient_in_cohort: int):
    diagnoses_df = pd.read_csv(diagnose_icd_file_path)

    if esrd:
        s_ids = filter_df_on_icd_code(diagnoses_df, esrd_codes, ckd_codes_stage3_to_5)
        print(
            f"Number of ESRD patients with CKD stage 3-5: {s_ids['subject_id'].nunique()}," 
            f"account for {s_ids['subject_id'].nunique()/num_patient_in_cohort*100:.3f} percent")

        s_ids = filter_df_on_icd_code(diagnoses_df, esrd_codes, ckd_codes_hypertension)
        print(
            f"Number of ESRD patients with hypertension: {s_ids['subject_id'].nunique()},"
            f"account for {s_ids['subject_id'].nunique() / num_patient_in_cohort*100:.3f} percent")

        s_ids = filter_df_on_icd_code(diagnoses_df, esrd_codes, ckd_codes_diabetes_mellitus)
        print(
            f"Number of ESRD patients with diabetes mellitus: {s_ids['subject_id'].nunique()},"
            f"account for {s_ids['subject_id'].nunique() / num_patient_in_cohort * 100:.3f} percent")
    else:
        s_ids = diagnoses_df[diagnoses_df['icd_code'].isin(ckd_codes_stage3_to_5)]
        print(
            f"Number of CKD patients with CKD stage 3-5: {s_ids['subject_id'].nunique()},"
            f"account for {s_ids['subject_id'].nunique() / num_patient_in_cohort * 100:.3f} percent")

        s_ids = diagnoses_df[diagnoses_df['icd_code'].isin(ckd_codes_hypertension)]
        print(
            f"Number of CKD patients with hypertension: {s_ids['subject_id'].nunique()},"
            f"account for {s_ids['subject_id'].nunique() / num_patient_in_cohort * 100:.3f} percent")

        s_ids = diagnoses_df[diagnoses_df['icd_code'].isin(ckd_codes_diabetes_mellitus)]
        print(
            f"Number of CKD patients with diabetes mellitus: {s_ids['subject_id'].nunique()},"
            f"account for {s_ids['subject_id'].nunique() / num_patient_in_cohort * 100:.3f} percent")


def analyze_ckd():
    patients_df, diagnoses_df = get_ckd_patients_and_diagnoses()
    plot_icd_codes(diagnoses_df)

    age_statistics(patients_df)
    gender_statistics(patients_df)
    ethnicity_and_race_statistics(patients_df, True, True)

    clinical_characteristic_analysis_esrd(esrd=False, num_patient_in_cohort=diagnoses_df['subject_id'].nunique())
    laboratory_params(patients_df)
    medication_use(patients_df)


def age_statistics(patients_df):
    print(
        f"age statistics:\n"
        f"mean: {patients_df['anchor_age'].mean():.3f}, std: {patients_df['anchor_age'].std():.3f}, min: {patients_df['anchor_age'].min()}, max: {patients_df['anchor_age'].max()}"
    )

    labels = ['<27', '27-54', '54-82', '82+']

    patients_df['age_group'] = pd.cut(patients_df['anchor_age'], bins=age_bins, labels=labels, right=False)

    vc = patients_df['age_group'].value_counts()
    vp = round(vc / len(patients_df) * 100, 3)
    res = pd.DataFrame({'Counts': vc, 'Percentage': vp})

    print(f"Distribution:\n{res}")


def plot_icd_codes(diagnoses_df):
    icd_code_counts = diagnoses_df['icd_code'].value_counts()

    plt.figure(figsize=(10, 7))
    icd_code_counts.plot.bar()

    # Add numbers on top of each bar
    for i, count in enumerate(icd_code_counts):
        plt.text(
            i,
            count,
            f'{count}',
            ha='center',
            va='bottom',
            fontsize=8
        )

    plt.title('Proportion Of ICD Codes (By Number of Diagnoses)')
    plt.ylabel('')  # Hide the y-label for the pie chart

    plt.savefig(figs_path_icd_stats, bbox_inches="tight")
    plt.clf()


def gender_statistics(patients_df):
    print(f"gender statistics:\n")

    vc = patients_df['gender'].value_counts()
    vp = round(vc / len(patients_df) * 100, 3)
    res = pd.DataFrame({'Counts': vc, 'Percentage': vp})

    print(f"Distribution:\n{res}")


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


# return race info
def ethnicity_and_race_statistics(patients_df, ethnicity_to_race: bool, verbose=False):
    admission_df = get_admission_df(ethnicity_to_race)
    admission_df = admission_df[admission_df['subject_id'].isin(patients_df['subject_id'])]
    admission_df = admission_df.drop_duplicates(subset='subject_id', keep='first')

    if verbose:
        vc = admission_df['race'].value_counts()
        vp = round(vc / len(admission_df) * 100, 3)
        res = pd.DataFrame({'Counts': vc, 'Percentage': vp})

        print(f"Distribution:\n{res}")

    return admission_df[['subject_id', 'race']]


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


# filter records for subjects for which there are records with 'icd_code' in arr_1 and arr_2
def filter_df_on_icd_code(df, arr_1, arr_2):
    subject_ids = df.groupby('subject_id').filter(
        lambda x:
        any(x['icd_code'].isin(arr_1)) and
        any(x['icd_code'].isin(arr_2))
    )['subject_id'].unique()
    return df[df['subject_id'].isin(subject_ids)]


if __name__ == '__main__':
    get_time_series_data_ckd_patients()
