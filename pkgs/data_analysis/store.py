import pandas as pd
from pkgs.commons import (ckd_codes, ckd_codes_stage3_to_5, diagnose_icd_file_path, esrd_codes, 
                          lab_events_file_path, lab_codes_creatinine, admissions_file_path, patients_file_path, 
                          lab_codes_proteins, lab_codes_albumin, lab_codes_potassium, lab_codes_urea_nitrogen, 
                          lab_codes_sodium, lab_codes_chloride, lab_codes_bicarbonate, lab_codes_anion_gap,
                          lab_codes_hematocrit, lab_codes_platelet_count, lab_codes_hemoglobin,
                          lab_codes_phosphate, lab_codes_calcium, lab_codes_glucose, lab_codes_serum_albumin,
                          lab_codes_wbc, lab_codes_rbc, lab_codes_mcv, lab_codes_mch, lab_codes_mchc,
                          lab_codes_rdw, lab_codes_magnesium, lab_codes_uric_acid, lab_codes_bilirubin_total,
                          lab_codes_alt, lab_codes_ast, lab_codes_alkaline_phosphatase, lab_codes_ldh,
                          lab_codes_iron, lab_codes_total_protein, lab_codes_cholesterol_total,
                          lab_codes_triglycerides, lab_codes_inr, lab_codes_ptt, lab_codes_crp,
                          lab_codes_ferritin, lab_codes_transferrin, lab_codes_tibc, lab_codes_lactate,
                          lab_codes_base_excess, lab_codes_pco2, lab_codes_po2, lab_codes_ph,
                          lab_codes_bilirubin_direct, lab_codes_bilirubin_indirect, lab_codes_ggt,
                          lab_codes_amylase, lab_codes_lipase, lab_codes_ck, lab_codes_troponin,
                          lab_codes_bnp, lab_codes_tsh, lab_codes_free_t4, lab_codes_vitamin_d,
                          lab_codes_pth, lab_codes_vitamin_b12, lab_codes_folate, lab_codes_reticulocyte,
                          lab_codes_fibrinogen, lab_codes_d_dimer, lab_codes_cortisol, lab_codes_hba1c,
                          lab_codes_ammonia, lab_codes_osmolality)
from pkgs.data_analysis.utils_store import filter_df_on_icd_code
from pkgs.data_analysis.utils import calculate_eGFR
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

    egfr_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_creatinine)]# unit is mg/dL
    egfr_df = pd.merge(egfr_df, patient_df, on='subject_id', how='outer')
    egfr_df = egfr_df[egfr_df['valuenum'] != 0]
    egfr_df['egfr'] = egfr_df.apply(calculate_eGFR, axis=1)

    egfr_df['serum_creatinine'] = egfr_df['valuenum']

    egfr_df['egfr'] = egfr_df['egfr'].replace('', np.nan)
    egfr_df = egfr_df.dropna(subset=['egfr'])

    return egfr_df

def get_protein_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_proteins)] # unit is mg/24hr
    lab_events_df['protein'] = lab_events_df['valuenum']

    lab_events_df['protein'] = lab_events_df['protein'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['protein'])
    return lab_events_df

def get_albumin_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_albumin)] # unit is mg/dL
    lab_events_df['albumin'] = lab_events_df['valuenum']

    lab_events_df['albumin'] = lab_events_df['albumin'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['albumin'])
    return lab_events_df

def get_potassium_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_potassium)] # unit is mEq/L
    lab_events_df['potassium'] = lab_events_df['valuenum']

    lab_events_df['potassium'] = lab_events_df['potassium'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['potassium'])
    return lab_events_df

def get_urea_nitrogen_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_urea_nitrogen)] # unit is mg/dL
    lab_events_df['urea_nitrogen'] = lab_events_df['valuenum']

    lab_events_df['urea_nitrogen'] = lab_events_df['urea_nitrogen'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['urea_nitrogen'])
    return lab_events_df

def get_sodium_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_sodium)] # unit is mEq/L
    lab_events_df['sodium'] = lab_events_df['valuenum']

    lab_events_df['sodium'] = lab_events_df['sodium'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['sodium'])
    return lab_events_df

def get_chloride_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_chloride)] # unit is mEq/L
    lab_events_df['chloride'] = lab_events_df['valuenum']

    lab_events_df['chloride'] = lab_events_df['chloride'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['chloride'])
    return lab_events_df

def get_bicarbonate_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_bicarbonate)] # unit is mEq/L
    lab_events_df['bicarbonate'] = lab_events_df['valuenum']

    lab_events_df['bicarbonate'] = lab_events_df['bicarbonate'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['bicarbonate'])
    return lab_events_df

def get_anion_gap_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_anion_gap)] # unit is mEq/L
    lab_events_df['anion_gap'] = lab_events_df['valuenum']

    lab_events_df['anion_gap'] = lab_events_df['anion_gap'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['anion_gap'])
    return lab_events_df

def get_hematocrit_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_hematocrit)] # unit is %
    lab_events_df['hematocrit'] = lab_events_df['valuenum']

    lab_events_df['hematocrit'] = lab_events_df['hematocrit'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['hematocrit'])
    return lab_events_df

def get_platelet_count_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_platelet_count)] # unit is K/uL
    lab_events_df['platelet_count'] = lab_events_df['valuenum']

    lab_events_df['platelet_count'] = lab_events_df['platelet_count'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['platelet_count'])
    return lab_events_df

def get_hemoglobin_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_hemoglobin)] # unit is g/dL
    lab_events_df['hemoglobin'] = lab_events_df['valuenum']

    lab_events_df['hemoglobin'] = lab_events_df['hemoglobin'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['hemoglobin'])
    return lab_events_df

def get_phosphate_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_phosphate)] # unit is mg/dL
    lab_events_df['phosphate'] = lab_events_df['valuenum']

    lab_events_df['phosphate'] = lab_events_df['phosphate'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['phosphate'])
    return lab_events_df

def get_calcium_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_calcium)] # unit is mg/dL
    lab_events_df['calcium'] = lab_events_df['valuenum']

    lab_events_df['calcium'] = lab_events_df['calcium'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['calcium'])
    return lab_events_df

def get_glucose_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_glucose)] # unit is mg/dL
    lab_events_df['glucose'] = lab_events_df['valuenum']

    lab_events_df['glucose'] = lab_events_df['glucose'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['glucose'])
    return lab_events_df

def get_serum_albumin_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_serum_albumin)] # unit is g/dL
    lab_events_df['serum_albumin'] = lab_events_df['valuenum']

    lab_events_df['serum_albumin'] = lab_events_df['serum_albumin'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['serum_albumin'])
    return lab_events_df

# Additional lab retrieval functions for CKD_THIRTY_FEATURES
def get_wbc_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_wbc)] # unit is K/uL
    lab_events_df['wbc'] = lab_events_df['valuenum']
    lab_events_df['wbc'] = lab_events_df['wbc'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['wbc'])
    return lab_events_df

def get_rbc_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_rbc)] # unit is M/uL
    lab_events_df['rbc'] = lab_events_df['valuenum']
    lab_events_df['rbc'] = lab_events_df['rbc'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['rbc'])
    return lab_events_df

def get_mcv_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_mcv)] # unit is fL
    lab_events_df['mcv'] = lab_events_df['valuenum']
    lab_events_df['mcv'] = lab_events_df['mcv'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['mcv'])
    return lab_events_df

def get_mch_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_mch)] # unit is pg
    lab_events_df['mch'] = lab_events_df['valuenum']
    lab_events_df['mch'] = lab_events_df['mch'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['mch'])
    return lab_events_df

def get_mchc_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_mchc)] # unit is g/dL
    lab_events_df['mchc'] = lab_events_df['valuenum']
    lab_events_df['mchc'] = lab_events_df['mchc'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['mchc'])
    return lab_events_df

def get_rdw_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_rdw)] # unit is %
    lab_events_df['rdw'] = lab_events_df['valuenum']
    lab_events_df['rdw'] = lab_events_df['rdw'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['rdw'])
    return lab_events_df

def get_magnesium_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_magnesium)] # unit is mg/dL
    lab_events_df['magnesium'] = lab_events_df['valuenum']
    lab_events_df['magnesium'] = lab_events_df['magnesium'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['magnesium'])
    return lab_events_df

def get_uric_acid_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_uric_acid)] # unit is mg/dL
    lab_events_df['uric_acid'] = lab_events_df['valuenum']
    lab_events_df['uric_acid'] = lab_events_df['uric_acid'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['uric_acid'])
    return lab_events_df

def get_bilirubin_total_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_bilirubin_total)] # unit is mg/dL
    lab_events_df['bilirubin_total'] = lab_events_df['valuenum']
    lab_events_df['bilirubin_total'] = lab_events_df['bilirubin_total'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['bilirubin_total'])
    return lab_events_df

def get_alt_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_alt)] # unit is IU/L
    lab_events_df['alt'] = lab_events_df['valuenum']
    lab_events_df['alt'] = lab_events_df['alt'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['alt'])
    return lab_events_df

def get_ast_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_ast)] # unit is IU/L
    lab_events_df['ast'] = lab_events_df['valuenum']
    lab_events_df['ast'] = lab_events_df['ast'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['ast'])
    return lab_events_df

def get_alkaline_phosphatase_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_alkaline_phosphatase)] # unit is IU/L
    lab_events_df['alkaline_phosphatase'] = lab_events_df['valuenum']
    lab_events_df['alkaline_phosphatase'] = lab_events_df['alkaline_phosphatase'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['alkaline_phosphatase'])
    return lab_events_df

def get_ldh_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_ldh)] # unit is IU/L
    lab_events_df['ldh'] = lab_events_df['valuenum']
    lab_events_df['ldh'] = lab_events_df['ldh'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['ldh'])
    return lab_events_df

def get_iron_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_iron)] # unit is ug/dL
    lab_events_df['iron'] = lab_events_df['valuenum']
    lab_events_df['iron'] = lab_events_df['iron'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['iron'])
    return lab_events_df

def get_total_protein_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_total_protein)] # unit is g/dL
    lab_events_df['total_protein'] = lab_events_df['valuenum']
    lab_events_df['total_protein'] = lab_events_df['total_protein'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['total_protein'])
    return lab_events_df

def get_cholesterol_total_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_cholesterol_total)] # unit is mg/dL
    lab_events_df['cholesterol_total'] = lab_events_df['valuenum']
    lab_events_df['cholesterol_total'] = lab_events_df['cholesterol_total'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['cholesterol_total'])
    return lab_events_df

def get_triglycerides_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_triglycerides)] # unit is mg/dL
    lab_events_df['triglycerides'] = lab_events_df['valuenum']
    lab_events_df['triglycerides'] = lab_events_df['triglycerides'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['triglycerides'])
    return lab_events_df

def get_inr_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_inr)] # ratio
    lab_events_df['inr'] = lab_events_df['valuenum']
    lab_events_df['inr'] = lab_events_df['inr'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['inr'])
    return lab_events_df

def get_ptt_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_ptt)] # unit is seconds
    lab_events_df['ptt'] = lab_events_df['valuenum']
    lab_events_df['ptt'] = lab_events_df['ptt'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['ptt'])
    return lab_events_df

def get_crp_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_crp)] # unit is mg/L
    lab_events_df['crp'] = lab_events_df['valuenum']
    lab_events_df['crp'] = lab_events_df['crp'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['crp'])
    return lab_events_df

# Additional lab retrieval functions for CKD_FIFTY_FEATURES_HETEROGENEOUS (labs 27-50)
def get_ferritin_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_ferritin)]
    lab_events_df['ferritin'] = lab_events_df['valuenum']
    lab_events_df['ferritin'] = lab_events_df['ferritin'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['ferritin'])
    return lab_events_df

def get_transferrin_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_transferrin)]
    lab_events_df['transferrin'] = lab_events_df['valuenum']
    lab_events_df['transferrin'] = lab_events_df['transferrin'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['transferrin'])
    return lab_events_df

def get_tibc_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_tibc)]
    lab_events_df['tibc'] = lab_events_df['valuenum']
    lab_events_df['tibc'] = lab_events_df['tibc'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['tibc'])
    return lab_events_df

def get_lactate_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_lactate)]
    lab_events_df['lactate'] = lab_events_df['valuenum']
    lab_events_df['lactate'] = lab_events_df['lactate'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['lactate'])
    return lab_events_df

def get_base_excess_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_base_excess)]
    lab_events_df['base_excess'] = lab_events_df['valuenum']
    lab_events_df['base_excess'] = lab_events_df['base_excess'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['base_excess'])
    return lab_events_df

def get_pco2_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_pco2)]
    lab_events_df['pco2'] = lab_events_df['valuenum']
    lab_events_df['pco2'] = lab_events_df['pco2'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['pco2'])
    return lab_events_df

def get_po2_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_po2)]
    lab_events_df['po2'] = lab_events_df['valuenum']
    lab_events_df['po2'] = lab_events_df['po2'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['po2'])
    return lab_events_df

def get_ph_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_ph)]
    lab_events_df['ph'] = lab_events_df['valuenum']
    lab_events_df['ph'] = lab_events_df['ph'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['ph'])
    return lab_events_df

def get_bilirubin_direct_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_bilirubin_direct)]
    lab_events_df['bilirubin_direct'] = lab_events_df['valuenum']
    lab_events_df['bilirubin_direct'] = lab_events_df['bilirubin_direct'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['bilirubin_direct'])
    return lab_events_df

def get_bilirubin_indirect_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_bilirubin_indirect)]
    lab_events_df['bilirubin_indirect'] = lab_events_df['valuenum']
    lab_events_df['bilirubin_indirect'] = lab_events_df['bilirubin_indirect'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['bilirubin_indirect'])
    return lab_events_df

def get_ggt_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_ggt)]
    lab_events_df['ggt'] = lab_events_df['valuenum']
    lab_events_df['ggt'] = lab_events_df['ggt'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['ggt'])
    return lab_events_df

def get_amylase_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_amylase)]
    lab_events_df['amylase'] = lab_events_df['valuenum']
    lab_events_df['amylase'] = lab_events_df['amylase'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['amylase'])
    return lab_events_df

def get_lipase_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_lipase)]
    lab_events_df['lipase'] = lab_events_df['valuenum']
    lab_events_df['lipase'] = lab_events_df['lipase'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['lipase'])
    return lab_events_df

def get_ck_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_ck)]
    lab_events_df['ck'] = lab_events_df['valuenum']
    lab_events_df['ck'] = lab_events_df['ck'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['ck'])
    return lab_events_df

def get_troponin_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_troponin)]
    lab_events_df['troponin'] = lab_events_df['valuenum']
    lab_events_df['troponin'] = lab_events_df['troponin'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['troponin'])
    return lab_events_df

def get_bnp_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_bnp)]
    lab_events_df['bnp'] = lab_events_df['valuenum']
    lab_events_df['bnp'] = lab_events_df['bnp'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['bnp'])
    return lab_events_df

def get_tsh_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_tsh)]
    lab_events_df['tsh'] = lab_events_df['valuenum']
    lab_events_df['tsh'] = lab_events_df['tsh'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['tsh'])
    return lab_events_df

def get_free_t4_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_free_t4)]
    lab_events_df['free_t4'] = lab_events_df['valuenum']
    lab_events_df['free_t4'] = lab_events_df['free_t4'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['free_t4'])
    return lab_events_df

def get_vitamin_d_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_vitamin_d)]
    lab_events_df['vitamin_d'] = lab_events_df['valuenum']
    lab_events_df['vitamin_d'] = lab_events_df['vitamin_d'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['vitamin_d'])
    return lab_events_df

def get_pth_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_pth)]
    lab_events_df['pth'] = lab_events_df['valuenum']
    lab_events_df['pth'] = lab_events_df['pth'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['pth'])
    return lab_events_df

def get_vitamin_b12_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_vitamin_b12)]
    lab_events_df['vitamin_b12'] = lab_events_df['valuenum']
    lab_events_df['vitamin_b12'] = lab_events_df['vitamin_b12'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['vitamin_b12'])
    return lab_events_df

def get_folate_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_folate)]
    lab_events_df['folate'] = lab_events_df['valuenum']
    lab_events_df['folate'] = lab_events_df['folate'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['folate'])
    return lab_events_df

def get_reticulocyte_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_reticulocyte)]
    lab_events_df['reticulocyte'] = lab_events_df['valuenum']
    lab_events_df['reticulocyte'] = lab_events_df['reticulocyte'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['reticulocyte'])
    return lab_events_df

def get_fibrinogen_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_fibrinogen)]
    lab_events_df['fibrinogen'] = lab_events_df['valuenum']
    lab_events_df['fibrinogen'] = lab_events_df['fibrinogen'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['fibrinogen'])
    return lab_events_df

def get_d_dimer_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_d_dimer)]
    lab_events_df['d_dimer'] = lab_events_df['valuenum']
    lab_events_df['d_dimer'] = lab_events_df['d_dimer'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['d_dimer'])
    return lab_events_df

def get_cortisol_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_cortisol)]
    lab_events_df['cortisol'] = lab_events_df['valuenum']
    lab_events_df['cortisol'] = lab_events_df['cortisol'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['cortisol'])
    return lab_events_df

def get_hba1c_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_hba1c)]
    lab_events_df['hba1c'] = lab_events_df['valuenum']
    lab_events_df['hba1c'] = lab_events_df['hba1c'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['hba1c'])
    return lab_events_df

def get_ammonia_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_ammonia)]
    lab_events_df['ammonia'] = lab_events_df['valuenum']
    lab_events_df['ammonia'] = lab_events_df['ammonia'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['ammonia'])
    return lab_events_df

def get_osmolality_df(patient_df):
    lab_events_df = get_lab_events_df_for_patients(patient_df)
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_osmolality)]
    lab_events_df['osmolality'] = lab_events_df['valuenum']
    lab_events_df['osmolality'] = lab_events_df['osmolality'].replace('', np.nan)
    lab_events_df = lab_events_df.dropna(subset=['osmolality'])
    return lab_events_df

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


def get_esrd_patients_and_diagnoses(diagnose_icd_file_path_supplied_s = '', patients_file_path_s = ''):
    if not diagnose_icd_file_path_supplied_s:
        diagnose_icd_file_path_supplied_s = diagnose_icd_file_path
    if not patients_file_path_s:
        patients_file_path_s = patients_file_path
    diagnoses_df = pd.read_csv(diagnose_icd_file_path_supplied_s)

    esrd_diagnose_df = filter_df_on_icd_code(diagnoses_df, esrd_codes, ckd_codes_stage3_to_5)
    esrd_diagnose_df = esrd_diagnose_df[esrd_diagnose_df['icd_code'].isin(esrd_codes)]
    print(
        f"number of ESRD subjects: {esrd_diagnose_df['subject_id'].nunique()}\n"
        f"percentage of subjects in dataset: {esrd_diagnose_df['subject_id'].nunique() / diagnoses_df['subject_id'].nunique() * 100:.3f}"
    )

    patients_df = pd.read_csv(patients_file_path_s)
    patients_df = patients_df[patients_df['subject_id'].isin(esrd_diagnose_df['subject_id'].unique())]

    print(f"number of subjects (for validation): {patients_df['subject_id'].nunique()}")

    return patients_df, esrd_diagnose_df

def get_ckd_but_non_esrd_patients_and_diagnoses():
    diagnoses_df = pd.read_csv(diagnose_icd_file_path)

    ckd_stage_35_diagnose_df = diagnoses_df[diagnoses_df['icd_code'].isin(ckd_codes_stage3_to_5)]
    print(
        f"number of CKD stage 3-5 subjects: {ckd_stage_35_diagnose_df['subject_id'].nunique()}\n"
        f"percentage of subjects in dataset: {ckd_stage_35_diagnose_df['subject_id'].nunique() / diagnoses_df['subject_id'].nunique() * 100:.3f}"
    )

    esrd_diagnose_df = diagnoses_df[diagnoses_df['icd_code'].isin(esrd_codes)]
    ckd_but_non_esrd_diagnose_df = ckd_stage_35_diagnose_df[~ckd_stage_35_diagnose_df['subject_id'].isin(esrd_diagnose_df['subject_id'])]

    patients_df = pd.read_csv(patients_file_path)
    patients_df = patients_df[patients_df['subject_id'].isin(ckd_but_non_esrd_diagnose_df['subject_id'].unique())]

    print(f"number of subjects (for validation): {patients_df['subject_id'].nunique()}")

    return patients_df, ckd_but_non_esrd_diagnose_df