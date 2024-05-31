import pandas as pd
from commons import (
    diagnose_icd_file_path, patients_file_path, get_kidney_failure_codes
)
def analysis_diagnose_icd():
    D = pd.read_csv(diagnose_icd_file_path)
    print(
        f"number of rows: {D.shape[0]}. number of subjects: {D['subject_id'].nunique()}"
    )

    print("filtering for kidney failure")
    k = get_kidney_failure_codes()
    df = D[D['icd_code'].isin(k)]
    print(
        f"number of rows: {df.shape[0]}. number of subjects: {df['subject_id'].nunique()}\n"
        f"percentage of subjects in dataset with kidney failure: {df['subject_id'].nunique() / D['subject_id'].nunique() * 100:.3f}"
    )

    kf_subjects = df['subject_id'].unique()
    print(f"number of subjects with kidney failure (for validation): {len(kf_subjects)}")

    df = pd.read_csv(patients_file_path)
    df = df[df['subject_id'].isin(kf_subjects)]

    print(
        f"printing number of subjects out for validation\n"
        f"number of rows: {df.shape[0]}. number of subjects: {df['subject_id'].nunique()}"
    )

    print("analyzing patients")

    # analyze gender statistics
    print(
        f"Gender statistics:\n"
        f"{df['gender'].value_counts()}."
    )


if __name__ == '__main__':
    analysis_diagnose_icd()