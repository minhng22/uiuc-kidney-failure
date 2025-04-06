import pandas as pd
from pkgs.commons import diagnose_icd_file_path, ckd_codes_stage3_to_5, esrd_codes
from pkgs.data.time_series_utils_store import calculate_duration_in_days
from pkgs.data.types import ExperimentScenario
from pkgs.commons import esrd_codes, patients_file_path
from pkgs.data.store import get_egfr_df, get_first_time_esrd_df, get_protein_df, get_albumin_df
import pandas as pd
from pkgs.data.types import ExperimentScenario
import numpy as np

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

    lab_df = calculate_duration_in_days(lab_df)

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

    return lab_df


# process patients who have not progressed to ESRD
def process_negative_patients(patient_ids: any, scenario_name: ExperimentScenario):
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
    
    lab_df = calculate_duration_in_days(lab_df)

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
    
    return lab_df

def get_lab_df_for_scenario_name(patients: any, scenario_name: ExperimentScenario):
    if scenario_name == ExperimentScenario.TIME_VARIANT:
        lab_df = get_egfr_df(patients)
    elif scenario_name == ExperimentScenario.HETEROGENEOUS:
        egfr_df = get_egfr_df(patients)
        egfr_df['egfr_missing'] = 0
        egfr_df['protein_missing'] = 1; egfr_df['protein'] = 0
        egfr_df['albumin_missing'] = 1; egfr_df['albumin'] = 0

        print('number of patients with egfr:', egfr_df['subject_id'].nunique())
        print('number of records with egfr:', len(egfr_df))

        protein_df = get_protein_df(patients)
        protein_df['protein_missing'] = 0
        protein_df['egfr_missing'] = 1; protein_df['egfr'] = 0
        protein_df['albumin_missing'] = 1; protein_df['albumin'] = 0

        print('number of patients with protein:', protein_df['subject_id'].nunique())
        print('number of records with protein:', len(protein_df))

        albumin_df = get_albumin_df(patients)
        albumin_df['albumin_missing'] = 0
        albumin_df['egfr_missing'] = 1; albumin_df['egfr'] = 0
        albumin_df['protein_missing'] = 1; albumin_df['protein'] = 0

        print('number of patients with albumin:', albumin_df['subject_id'].nunique())
        print('number of records with albumin:', len(albumin_df))
        
        lab_df = pd.concat([egfr_df, protein_df, albumin_df])
    else:
        assert scenario_name == ExperimentScenario.EGFR_COMPONENTS, f"Unknown scenario name: {scenario_name}"
        lab_df = get_egfr_df(patients)
        lab_df['gender'] = lab_df['gender'].map({'M': 1, 'F': 0})
    
    lab_df.rename(columns={'anchor_age': 'age', 'charttime': 'time'}, inplace=True)

    lab_df[get_feature_columns(scenario_name)] = lab_df[get_feature_columns(scenario_name)].replace('', np.nan)
    lab_df.dropna(subset=get_feature_columns(scenario_name), inplace=True)

    print('Finished getting raw lab records for scenario:', scenario_name)
    print(lab_df.head())
    print(lab_df.columns.tolist())

    return lab_df

def get_feature_columns(scenario):
    if scenario == ExperimentScenario.TIME_VARIANT:
        return ['egfr']
    elif scenario == ExperimentScenario.HETEROGENEOUS:
        return ['egfr', 'protein', 'albumin']
    elif scenario == ExperimentScenario.EGFR_COMPONENTS:
        return ['age', 'gender', 'serum_creatinine']

def add_time_variant_support(df):
    df = df.sort_values(by=['subject_id', 'duration_in_days'])

    df['start'] = df['duration_in_days']
    df['stop'] = df.groupby('subject_id')['start'].shift(-1) + 1e-5
    df['stop'] = df['stop'].fillna(df['start'] + 1e-5)

    df.reset_index(drop=True, inplace=True)
    print('add_time_variant_support cols: ', df.columns.tolist())

    # validate start < stop
    invalid_rows = df[df['stop'] < df['start']]
    if not invalid_rows.empty:
        print("Subjects with 'stop' < 'start':")
        print(invalid_rows['subject_id'].tolist())
        raise ValueError("Invalid data: 'stop' < 'start' for some subjects.")
    else:
        print("No subjects found with 'stop' < 'start'.")
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
    
    lab_df.reset_index(drop=True, inplace=True)

    print(f"Data: \n{lab_df.head()}\n"
          f"Total number of patients: {lab_df['subject_id'].nunique()}\n"
          f"Total number of records: {len(lab_df)}")
    return lab_df[get_final_columns(scenario)]

def get_final_columns(scenario):
    if scenario == ExperimentScenario.TIME_INVARIANT:
        return ['subject_id', 'duration_in_days', 'egfr', 'has_esrd']
    elif scenario == ExperimentScenario.TIME_VARIANT:
        return ['subject_id', 'duration_in_days', 'start', 'stop', 'egfr', 'has_esrd']
    elif scenario == ExperimentScenario.HETEROGENEOUS:
        return ['subject_id', 'duration_in_days', 'start', 'stop', 'egfr', 'egfr_missing', 'protein', 'protein_missing', 'albumin', 'albumin_missing', 'has_esrd']
    elif scenario == ExperimentScenario.EGFR_COMPONENTS:
        return ['subject_id', 'duration_in_days', 'start', 'stop', 'age', 'gender', 'serum_creatinine', 'has_esrd']
   
if __name__ == '__main__':
    # get_time_series_data_ckd_patients('egfr_components')
    data = get_time_series_data_ckd_patients(ExperimentScenario.TIME_VARIANT)
    # Select rows that contain NaN values
    rows_with_nan = data[data.isnull().any(axis=1)]
    print("Rows with NaN values:\n%s", rows_with_nan)
