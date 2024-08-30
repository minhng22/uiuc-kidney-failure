# No records of albumin for patients progressed from ckd 3-5 to esrd.
from pkgs.commons import chart_events_file_path, lab_codes_albumin, lab_events_file_path


import pandas as pd


def verify_that_albumin_records_not_exist_for_patients(patient_ids):
    lab_events_df = pd.read_csv(lab_events_file_path)
    lab_albumin_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_albumin)]
    print(f'Number of records for albumin: {len(lab_albumin_df)}')

    chart_events_df = pd.read_csv(chart_events_file_path)
    chart_events_df = chart_events_df[chart_events_df['subject_id'].isin(patient_ids)]

    chart_albumin_df = chart_events_df[chart_events_df['itemid'].isin(lab_codes_albumin)]
    print(f'Number of records for albumin: {len(chart_albumin_df)}')


def eval_duration(df):
    esrd_subjects_df = df[df['has_esrd'] == True]
    max_duration = esrd_subjects_df['duration_in_days'].max()
    print(f"max duration: {max_duration}")