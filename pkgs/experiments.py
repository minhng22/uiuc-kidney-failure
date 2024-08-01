import pandas as pd

from pkgs.commons import esrd_codes, ckd_codes_stage3_to_5, lab_codes_egfr, lab_events_file_path, lab_codes_albumin, \
    chart_events_file_path, diagnose_icd_file_path, patients_file_path
from lifelines import CoxPHFitter

from pkgs.data import filter_df_on_icd_code


# No records of albumin for patients progressed from ckd 3-5 to esrd.
def verify_that_albumin_records_not_exist_for_patients(patient_ids):
    lab_events_df = pd.read_csv(lab_events_file_path)
    lab_albumin_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_albumin)]
    print(f'Number of records for albumin: {len(lab_albumin_df)}')

    chart_events_df = pd.read_csv(chart_events_file_path)
    chart_events_df = chart_events_df[chart_events_df['subject_id'].isin(patient_ids)]

    chart_albumin_df = chart_events_df[chart_events_df['itemid'].isin(lab_codes_albumin)]
    print(f'Number of records for albumin: {len(chart_albumin_df)}')

def run_cox_model():
    diagnoses_df = pd.read_csv(diagnose_icd_file_path)
    patient_ids = filter_df_on_icd_code(
        diagnoses_df, esrd_codes, ckd_codes_stage3_to_5)['subject_id'].unique()
    print(f'Number of patients progressed from ckd stage 3-5 to esrd are {len(patient_ids)}')

    patients = pd.read_csv(patients_file_path)
    patients = patients[patients['subject_id'].isin(patient_ids)]
    print(f'Number of dead patients are {patients['dod'].notna().sum()} '
          f'out of total {len(patients)}')

    lab_events_df = pd.read_csv(lab_events_file_path)
    lab_events_df = lab_events_df[lab_events_df['subject_id'].isin(patient_ids)]
    lab_events_df['itemid'] = lab_events_df['itemid'].astype(str)
    print(f'Number of lab events {len(lab_events_df)}')
    lab_events_df = lab_events_df[lab_events_df['itemid'].isin(lab_codes_egfr)]
    lab_events_df = lab_events_df[lab_events_df['valuenum'].notnull()]
    print(f'Number of eGFR events {len(lab_events_df)}')
    print(lab_events_df.columns.tolist())

    # Merging the two DataFrames on 'subject_id'
    data = pd.merge(lab_events_df, patients, on='subject_id', how='outer')
    data['eGFR'] = data['valuenum']
    data['dead'] = data['dod'].notna().astype(int)
    data['age'] = data['anchor_age']

    # 'gender' has low variance and affects convergence
    data = data[['subject_id', 'age', 'eGFR', 'dead']]

    print(data)

    # # Initialize the CoxPHFitter
    # cph = CoxPHFitter()
    #
    # # Fit the model
    # cph.fit(data, duration_col='age', event_col='dead', show_progress=True)
    #
    # # Print the summary
    # cph.print_summary()
    #
    # M = cph.predict_median(data[data['dead'] == 0])
    # print(f"predict_median\n{M}")


if __name__ == '__main__':
    run_cox_model()