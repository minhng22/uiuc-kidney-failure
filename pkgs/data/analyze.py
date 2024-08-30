from pkgs.data.df_supply import get_esrd_patients_and_diagnoses, \
    get_admission_df, get_lab_events_for_patients, get_egfr_df
from pkgs.data.patients_and_diagnosis_supply import get_ckd_patients_and_diagnoses
from pkgs.data.graphic import plot_icd_codes
from commons import lab_codes_creatinine, esrd_codes, ckd_codes_stage3_to_5, ckd_codes_hypertension, \
    ckd_codes_diabetes_mellitus, ace_inhibitor_drugs, diagnose_icd_file_path, age_bins, prescription_file_path
from pkgs.data.df_process import filter_df_on_icd_code
import pandas as pd


def add_race_to_patients(patient_df, verbose:bool=False):
    race_df = ethnicity_and_race_statistics(patient_df, True)
    patient_df = pd.merge(patient_df, race_df, on='subject_id', how='outer')
    if verbose:
        print(f"Patients df with race: \n{patient_df[['subject_id', 'race', 'gender', 'anchor_age']].head()}")

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


def analyze_esrd():
    patients_df, diagnoses_df = get_esrd_patients_and_diagnoses()

    #plot_icd_codes(diagnoses_df)
    # age_statistics(patients_df)
    # gender_statistics(patients_df)
    # ethnicity_and_race_statistics(patients_df, True, True)

    #clinical_characteristic_analysis_esrd(esrd=True, num_patient_in_cohort=diagnoses_df['subject_id'].nunique())
    laboratory_params(patients_df)
    #medication_use(patients_df)


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

def gender_statistics(patients_df):
    print(f"gender statistics:\n")

    vc = patients_df['gender'].value_counts()
    vp = round(vc / len(patients_df) * 100, 3)
    res = pd.DataFrame({'Counts': vc, 'Percentage': vp})

    print(f"Distribution:\n{res}")


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


def analyze_ckd():
    patients_df, diagnoses_df = get_ckd_patients_and_diagnoses()
    plot_icd_codes(diagnoses_df)

    age_statistics(patients_df)
    gender_statistics(patients_df)
    ethnicity_and_race_statistics(patients_df, True, True)

    clinical_characteristic_analysis_esrd(esrd=False, num_patient_in_cohort=diagnoses_df['subject_id'].nunique())
    laboratory_params(patients_df)
    medication_use(patients_df)