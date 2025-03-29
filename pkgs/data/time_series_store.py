import pandas as pd
from pkgs.commons import diagnose_icd_file_path, ckd_codes_stage3_to_5, esrd_codes
from pkgs.data.time_series_utils_store import process_negative_patients, process_positive_patients
import numpy as np
from pkgs.data.types import ExperimentScenario

def add_time_variant_support(df):
    df = df.sort_values(by=['subject_id', 'duration_in_days'])

    df['start'] = df['duration_in_days']
    df['stop'] = df.groupby('subject_id')['duration_in_days'].shift(-1) - 1e-5
    df['stop'] = df['stop'].fillna(df['start'] + 1e-5)

    df.reset_index(drop=True, inplace=True)
    return df


# get late stage ckd patients and info of their progression to esrd.
# only_esrd set to True returns only patients who have progressed to ESRD.
# there are four scenarios:
# 1. 'time_invariant'
# 2. 'time_variant'
# 3. 'heterogeneous': a variation of time_variant where the lab measurements are contains [egfr, proteinuria]
# 4. 'egfr_components': a variation of time_variant where the features are components of egfr [age, sex, serum_creatinine]
def get_time_series_data_ckd_patients(scenario: ExperimentScenario):
    diagnoses_df = pd.read_csv(diagnose_icd_file_path)
    diagnoses_df = diagnoses_df[diagnoses_df['icd_code'].isin(ckd_codes_stage3_to_5 + esrd_codes)]
    diagnoses_df.dropna()

    esrd_patients = diagnoses_df[diagnoses_df['icd_code'].isin(esrd_codes)]['subject_id'].unique()
    non_esrd_patients = diagnoses_df[~diagnoses_df['subject_id'].isin(esrd_patients)]['subject_id'].unique()
    print(f"Sample patients with esrd: {esrd_patients[:10]}")
    print(f'Number of patients progressed from ckd stage 3-5 to esrd are {len(esrd_patients)} '
          f'over {diagnoses_df["subject_id"].nunique()}, '
          f'accounts for {round(100 * len(esrd_patients)/diagnoses_df["subject_id"].nunique(), 3)}%')
    print(f'Number of patients who have not progressed to esrd are {len(non_esrd_patients)} '
          f'over {diagnoses_df["subject_id"].nunique()}, '
          f'accounts for {round(100 * len(non_esrd_patients)/diagnoses_df["subject_id"].nunique(), 3)}%')
    
    lab_df_1 = process_negative_patients(non_esrd_patients, scenario)
    lab_df_2 = process_positive_patients(diagnoses_df, esrd_patients, scenario)

    lab_df = pd.concat([lab_df_1, lab_df_2])
    print(f"After merge:\n"
          f"Total number of patients: {lab_df['subject_id'].nunique()}\n"
          f"Total number of records: {len(lab_df)}")
    
    lab_df['subject_id'] = lab_df['subject_id'].astype(int)
    lab_df['duration_in_days'] = lab_df['duration_in_days'].astype(float)

    if scenario == ExperimentScenario.TIME_INVARIANT:
        # right-censoring. similar to work done by:
        # 1. Hagar et al.: Survival Analysis of EHR CKD Data
        d = pd.DataFrame(columns=lab_df.columns)
        for _, group in lab_df.groupby('subject_id'):
            max_row = group.loc[group['duration_in_days'].idxmax()]
            d = d._append(max_row)
        lab_df = d
    elif scenario == ExperimentScenario.TIME_VARIANT:
        lab_df = add_time_variant_support(lab_df)[['subject_id', 'duration_in_days', 'start', 'stop', 'egfr', 'has_esrd']]
    elif scenario == ExperimentScenario.HETEROGENEOUS:
        lab_df = add_time_variant_support(lab_df)[['subject_id', 'duration_in_days', 'start', 'stop', 'egfr', 'egfr_missing', 'protein', 'protein_missing', 'albumin', 'albumin_missing', 'has_esrd']]
    elif scenario == ExperimentScenario.EGFR_COMPONENTS:
        lab_df = add_time_variant_support(lab_df)[['subject_id', 'duration_in_days', 'start', 'stop', 'age', 'gender', 'serum_creatinine', 'has_esrd']]
    
    lab_df.dropna(inplace=True)
    lab_df = lab_df.replace('', np.nan).dropna()
    
    lab_df.reset_index(drop=True, inplace=True)

    print(f"Data: \n{lab_df.head()}\n"
          f"Total number of patients: {lab_df['subject_id'].nunique()}\n"
          f"Total number of records: {len(lab_df)}")
    return lab_df

if __name__ == '__main__':
    # get_time_series_data_ckd_patients('egfr_components')
    get_time_series_data_ckd_patients(ExperimentScenario.HETEROGENEOUS)
