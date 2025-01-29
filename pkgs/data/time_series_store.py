import pandas as pd
from pkgs.commons import diagnose_icd_file_path, ckd_codes_stage3_to_5, esrd_codes
from pkgs.data.time_series_utils_store import process_negative_patients, process_positive_patients


def prep_data(df):
    df['start'] = df.groupby('subject_id').cumcount() * df['duration_in_days']  # Calculate start times
    df['stop'] = df['start'] + df['duration_in_days']  # Calculate stop times
    df.dropna(inplace=True)
    # Adjust rows where start == stop by adding a small value to stop
    df.loc[df['start'] == df['stop'], 'stop'] += 1e-5  # Adding a small value to ensure stop > start
    df.reset_index(drop=True, inplace=True)
    return df


# get late stage ckd patients and info of their progression to esrd.
# only_esrd set to True returns only patients who have progressed to ESRD.
def get_time_series_data_ckd_patients(time_variant, multiple_risk = False):
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
    
    lab_df_1 = process_negative_patients(non_esrd_patients, multiple_risk)
    lab_df_2 = process_positive_patients(diagnoses_df, esrd_patients, multiple_risk)

    lab_df = pd.concat([lab_df_1, lab_df_2])
    print(f"After merge:\n"
          f"Total number of patients: {lab_df['subject_id'].nunique()}\n"
          f"Total number of records: {len(lab_df)}")
    
    lab_df['subject_id'] = lab_df['subject_id'].astype(int)
    lab_df['duration_in_days'] = lab_df['duration_in_days'].astype(float)
    
    if time_variant:
        lab_df = prep_data(lab_df)[['subject_id', 'duration_in_days', 'start', 'stop', 'has_esrd', 'egfr']]
    else:
        # right-censoring. similar to work done by:
        # 1. Hagar et al.: Survival Analysis of EHR CKD Data
        d = pd.DataFrame(columns=lab_df.columns)
        for _, group in lab_df.groupby('subject_id'):
            max_row = group.loc[group['duration_in_days'].idxmax()]
            d = d._append(max_row)
        lab_df = d
        lab_df.dropna(inplace=True)
        lab_df.reset_index(drop=True, inplace=True)

    print(f"Data: \n{lab_df.head()}\n"
          f"Total number of patients: {lab_df['subject_id'].nunique()}\n"
          f"Total number of records: {len(lab_df)}")
    return lab_df
