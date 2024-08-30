import pandas as pd
from pkgs.commons import admissions_file_path, esrd_codes


# filter records for subjects for which there are records with 'icd_code' in arr_1 and arr_2
def filter_df_on_icd_code(df, arr_1, arr_2):
    subject_ids = df.groupby('subject_id').filter(
        lambda x:
        any(x['icd_code'].isin(arr_1)) and
        any(x['icd_code'].isin(arr_2))
    )['subject_id'].unique()
    return df[df['subject_id'].isin(subject_ids)]


def get_first_time_esrd(diagnose_df):
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