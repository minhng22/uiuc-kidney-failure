import pandas as pd
from pkgs.commons import diagnose_icd_file_path, ckd_codes_stage3_to_5, esrd_codes, patients_file_path
from pkgs.data.df_supply import get_egfr_df
from pkgs.data.df_process import get_first_time_esrd


# get late stage ckd patients and info of their progression to esrd.
# only_esrd set to True returns only patients who have progressed to ESRD.
def get_time_series_data_ckd_patients():
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
    
    lab_df_1 = process_negative_patients(non_esrd_patients)
    lab_df_2 = process_positive_patients(diagnoses_df, esrd_patients)

    lab_df = pd.concat([lab_df_1, lab_df_2])
    print(f"Final data: \n{lab_df.head()}\n"
          f"Total number of patients: {lab_df['subject_id'].nunique()}\n")

    return lab_df


# process patients who have not progressed to ESRD
def process_negative_patients(patient_ids):
    print(
        f"Processing patients who have not progressed to ESRD:\n"
        f"Number of patients: {len(patient_ids)}")
    
    patients = pd.read_csv(patients_file_path)
    patients = patients[patients['subject_id'].isin(patient_ids)]
    lab_df = get_egfr_df(patients)

    lab_df.rename(columns={'anchor_age': 'age', 'charttime': 'time'}, inplace=True)
    lab_df['has_esrd'] = 0

    lab_df['time'] = pd.to_datetime(lab_df['time'])
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

    # keep patients who have at least 2 records.
    lab_df = lab_df.groupby('subject_id').filter(lambda x: len(x) >= 2)
    print(
        f"Number of patients after filtering out < 2 records: {lab_df['subject_id'].nunique()}\n"
        f"Records sample:\n{lab_df[['subject_id', 'time', 'has_esrd']].head()}")

    print(
        f"Stats on eGFR:\n"
        f"Number of records: {len(lab_df)}. Number of patients: {lab_df['subject_id'].nunique()}\n"
        f"mean {lab_df['egfr'].mean():.3f} sd {lab_df['egfr'].std():.3f}")
    return lab_df[['subject_id', 'duration_in_days', 'egfr', 'has_esrd']]


# process patients who have progressed to ESRD
def process_positive_patients(diagnoses_df, patient_ids):
    print(
        f"Processing patients who have progressed to ESRD:\n"
        f"Number of patients: {len(patient_ids)}")
    diagnose_esrd_df = diagnoses_df[diagnoses_df['icd_code'].isin(esrd_codes)]
    print(f"Number of patients with ESRD: {diagnose_esrd_df['subject_id'].nunique()}")
    first_time_esrd_df = get_first_time_esrd(diagnose_esrd_df)

    patients = pd.read_csv(patients_file_path)
    patients = patients[patients['subject_id'].isin(patient_ids)]
    lab_df = get_egfr_df(patients)

    lab_df.rename(columns={'anchor_age': 'age', 'charttime': 'time'}, inplace=True)
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

    # keep patients who have at least 2 records.
    lab_df = lab_df.groupby('subject_id').filter(lambda x: len(x) >= 2)
    print(
        f"Number of patients after filtering out < 2 records: {lab_df['subject_id'].nunique()}\n"
        f"Records sample:\n{lab_df[['subject_id', 'time', 'has_esrd', 'first_diagnose_esrd_time']].head()}")
    
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
        
        # Group by 'subject_id' and filter groups with fewer than 2 rows
        subjects_less_than_2_rows = D.groupby('subject_id').filter(lambda x: len(x) < 2)

        # Get the unique subject IDs with fewer than 2 rows
        for patient in subjects_less_than_2_rows['subject_id'].unique()[:5]:
            print(
                f"subject_id: {patient} is bad.\n"
                f"data: {subjects_less_than_2_rows[subjects_less_than_2_rows['subject_id'] == patient][['subject_id', 'time', 'has_esrd', 'first_diagnose_esrd_time']]}\n")
    
    validate(lab_df)

    print(
        f"Stats on eGFR:\n"
        f"Number of records: {len(lab_df)}. Number of patients: {lab_df['subject_id'].nunique()}\n"
        f"mean {lab_df['egfr'].mean():.3f} sd {lab_df['egfr'].std():.3f}")
    
    return lab_df[['subject_id', 'duration_in_days', 'egfr', 'has_esrd']]