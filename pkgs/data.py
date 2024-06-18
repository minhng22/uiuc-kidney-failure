import re

import matplotlib.pyplot as plt
import pandas as pd

from commons import (
    diagnose_icd_file_path, patients_file_path,
    age_bins, esrd_codes,
    ckd_codes, admissions_file_path, ckd_codes_stage3_to_5, ckd_codes_hypertension, ckd_codes_diabetes_mellitus,
    lab_events_file_path, creatinine_lab_codes, proteins_24hr_lab_codes, omr_file_path,
    prescription_file_path, ace_inhibitor_drugs, figs_path_icd_stats
)


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
    plot_icd_codes(diagnoses_df)

    age_statistics(patients_df)
    gender_statistics(patients_df)

    clinical_characteristic_analysis_esrd(esrd=True, num_patient_in_cohort=diagnoses_df['subject_id'].nunique())
    laboratory_params(patients_df)
    medication_use(patients_df)


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

def laboratory_params(patient_df):
    lab_events_df = pd.read_csv(lab_events_file_path)
    lab_events_df['itemid'] = lab_events_df['itemid'].astype(str)
    lab_events_df['valuenum'] = lab_events_df['valuenum'].astype(float)

    # serum creatinine
    sc_df = lab_events_df[lab_events_df['itemid'].isin(creatinine_lab_codes)]
    print(f"Number of records for Serum Creatinine: {len(sc_df)}")
    sc_df = sc_df[sc_df['subject_id'].isin(patient_df['subject_id'])]
    print(f"units: {sc_df['valueuom'].value_counts()}")
    print(
        f"Stats on Serum Creatinine:\n" 
        f"Number of records: {len(sc_df)}\n"
        f"mean {sc_df['valuenum'].mean():.3f} sd {sc_df['valuenum'].std():.3f}")

    # eGFR
    # No eGFR value for eGFR records in labevents.csv
    # Per MIMIC-IV release note https://physionet.org/content/mimiciv/2.2/#files-panel, we can get this value from omr.csv
    # uom of eGFR is (mL/min/1.73 m2) (check labevents.csv for eGFR code to validate this)
    omr_df = pd.read_csv(omr_file_path)
    omr_df = omr_df[omr_df['subject_id'].isin(patient_df['subject_id'])]
    omr_df = omr_df[omr_df['result_name'] == 'eGFR']

    omr_df['result_value'] = omr_df['result_value'].apply(extract_num_from_value)

    print(f"Number of records for eGFR: {len(omr_df)}")
    print(
        f"Stats on eGFR:\n"
        f"Number of records: {len(omr_df)}\n"
        f"mean {omr_df['result_value'].mean():.3f} sd {omr_df['result_value'].std():.3f}")

    # 24hr urine protein
    protein_24hr_df = lab_events_df[lab_events_df['itemid'].isin(proteins_24hr_lab_codes)]
    print(f"Number of records for 24hr urine protein: {len(protein_24hr_df)}")
    protein_24hr_df = protein_24hr_df[protein_24hr_df['subject_id'].isin(patient_df['subject_id'])]
    protein_24hr_df['valuenum'] = protein_24hr_df['valuenum'] / 1000 # mg/24hr to g/24hr

    print(f"units: {protein_24hr_df['valueuom'].value_counts()}")
    print(
        f"Stats on 24hr urine protein:\n"
        f"Number of records: {len(protein_24hr_df)}\n"
        f"median {protein_24hr_df['valuenum'].median():.3f} IQR {(protein_24hr_df['valuenum'].quantile(0.75) - protein_24hr_df['valuenum'].quantile(0.25)):.3f}")


def clinical_characteristic_analysis_esrd(esrd: bool, num_patient_in_cohort: int):
    diagnoses_df = pd.read_csv(diagnose_icd_file_path)

    if esrd:
        s_ids = filter_diagnoses_for_patients_with_both_icd_codes(diagnoses_df, esrd_codes, ckd_codes_stage3_to_5)
        print(
            f"Number of ESRD patients with CKD stage 3-5: {s_ids['subject_id'].nunique()}," 
            f"account for {s_ids['subject_id'].nunique()/num_patient_in_cohort*100:.3f} percent")

        s_ids = filter_diagnoses_for_patients_with_both_icd_codes(diagnoses_df, esrd_codes, ckd_codes_hypertension)
        print(
            f"Number of ESRD patients with hypertension: {s_ids['subject_id'].nunique()},"
            f"account for {s_ids['subject_id'].nunique() / num_patient_in_cohort*100:.3f} percent")

        s_ids = filter_diagnoses_for_patients_with_both_icd_codes(diagnoses_df, esrd_codes, ckd_codes_diabetes_mellitus)
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
    ethnicity_and_race_statistics(patients_df, True)
    ethnicity_and_race_statistics(patients_df, False)

    # clinical_characteristic_analysis_esrd(esrd=False, num_patient_in_cohort=diagnoses_df['subject_id'].nunique())
    # laboratory_params(patients_df)
    # medication_use(patients_df)


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

    print(f"percentage of patients with race selection 'PATIENT DECLINED TO ANSWER', "
          f"'UNABLE TO OBTAIN', or 'UNKNOWN': {percentage_filtered:.2f}%")
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


def ethnicity_and_race_statistics(patients_df, ethnicity_to_race: bool):
    admission_df = get_admission_df(ethnicity_to_race)
    admission_df = admission_df[admission_df['subject_id'].isin(patients_df['subject_id'])]

    vc = admission_df['race'].value_counts()
    vp = round(vc / len(admission_df) * 100, 3)
    res = pd.DataFrame({'Counts': vc, 'Percentage': vp})

    print(f"Distribution:\n{res}")


def get_esrd_patients_and_diagnoses():
    diagnoses_df = pd.read_csv(diagnose_icd_file_path)

    esrd_diagnose_df = filter_diagnoses_for_patients_with_both_icd_codes(diagnoses_df, esrd_codes, ckd_codes)
    esrd_diagnose_df = esrd_diagnose_df[esrd_diagnose_df['icd_code'].isin(esrd_codes)]
    print(
        f"number of ESRD subjects: {esrd_diagnose_df['subject_id'].nunique()}\n"
        f"percentage of subjects in dataset: {esrd_diagnose_df['subject_id'].nunique() / diagnoses_df['subject_id'].nunique() * 100:.3f}"
    )

    patients_df = pd.read_csv(patients_file_path)
    patients_df = patients_df[patients_df['subject_id'].isin(esrd_diagnose_df['subject_id'].unique())]

    print(f"number of subjects (for validation): {patients_df['subject_id'].nunique()}")

    return patients_df, esrd_diagnose_df


def get_ckd_patients_and_diagnoses():
    diagnoses_df = pd.read_csv(diagnose_icd_file_path)

    ckd_diagnose_df = diagnoses_df[diagnoses_df['icd_code'].isin(ckd_codes)]
    print(
        f"number of CKD subjects: {ckd_diagnose_df['subject_id'].nunique()}\n"
        f"percentage of subjects in dataset: {ckd_diagnose_df['subject_id'].nunique() / diagnoses_df['subject_id'].nunique() * 100:.3f}"
    )

    patients_df = pd.read_csv(patients_file_path)
    patients_df = patients_df[patients_df['subject_id'].isin(ckd_diagnose_df['subject_id'].unique())]

    print(f"number of subjects (for validation): {patients_df['subject_id'].nunique()}")

    return patients_df, ckd_diagnose_df


# filter records for subjects for which there are records with 'icd_code' in arr_1 and arr_2
def filter_diagnoses_for_patients_with_both_icd_codes(df, arr_1, arr_2):
    subject_ids = df.groupby('subject_id').filter(
        lambda x:
        any(x['icd_code'].isin(arr_1)) and
        any(x['icd_code'].isin(arr_2))
    )['subject_id'].unique()
    return df[df['subject_id'].isin(subject_ids)]


if __name__ == '__main__':
    analyze_ckd()
