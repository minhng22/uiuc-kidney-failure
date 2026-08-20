"""
One-off script for the CKD Fifty Features mini experiment.

Builds a small, stratified (by ESRD status) random subsample of the
CKD_FIFTY_FEATURES_HETEROGENEOUS train/test data from rep1, and writes it to an
isolated rep99 slot so it can be used with `CKD_REP=99` without touching rep1-5's
live data or model files.

See CKD_FIFTY_FEATURES_mini_experiment_plan.md for the full plan.
"""
import os
import numpy as np
import pandas as pd

from pkgs.commons import project_dir

RANDOM_SEED = 42
PATIENTS_PER_CLASS = 250  # 250 ESRD + 250 non-ESRD = 500 total per split

SRC_REP = 1
DST_REP = 99

SRC_DIR = f"{project_dir()}/generated_data/rep{SRC_REP}"
DST_DIR = f"{project_dir()}/generated_data/rep{DST_REP}"

TRAIN_SRC = f"{SRC_DIR}/ckd_fifty_features_heterogeneous_train_data.csv"
TEST_SRC = f"{SRC_DIR}/ckd_fifty_features_heterogeneous_test_data.csv"
ESRD_IDS_SRC = f"{SRC_DIR}/esrd_patient_ids.csv"

TRAIN_DST = f"{DST_DIR}/ckd_fifty_features_heterogeneous_train_data.csv"
TEST_DST = f"{DST_DIR}/ckd_fifty_features_heterogeneous_test_data.csv"


def stratified_sample(df: pd.DataFrame, esrd_patient_ids: np.ndarray, rng: np.random.Generator, label: str) -> pd.DataFrame:
    esrd_subjects = df[df["subject_id"].isin(esrd_patient_ids)]["subject_id"].unique()
    non_esrd_subjects = df[~df["subject_id"].isin(esrd_patient_ids)]["subject_id"].unique()

    n_esrd = min(PATIENTS_PER_CLASS, len(esrd_subjects))
    n_non_esrd = min(PATIENTS_PER_CLASS, len(non_esrd_subjects))

    picked_esrd = rng.choice(esrd_subjects, size=n_esrd, replace=False)
    picked_non_esrd = rng.choice(non_esrd_subjects, size=n_non_esrd, replace=False)

    picked = np.concatenate([picked_esrd, picked_non_esrd])
    sampled = df[df["subject_id"].isin(picked)].reset_index(drop=True)

    print(
        f"[{label}] source patients: {df['subject_id'].nunique()} "
        f"(esrd={len(esrd_subjects)}, non_esrd={len(non_esrd_subjects)}) -> "
        f"sampled patients: {sampled['subject_id'].nunique()} "
        f"(esrd={n_esrd}, non_esrd={n_non_esrd}), rows: {len(sampled)}"
    )
    return sampled


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    esrd_ids = pd.read_csv(ESRD_IDS_SRC)["subject_id"].unique()
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Reading train data from {TRAIN_SRC}")
    train_df = pd.read_csv(TRAIN_SRC, index_col=0)
    train_sample = stratified_sample(train_df, esrd_ids, rng, "train")
    train_sample.to_csv(TRAIN_DST)
    print(f"Wrote {TRAIN_DST}")
    del train_df, train_sample

    print(f"Reading test data from {TEST_SRC}")
    test_df = pd.read_csv(TEST_SRC, index_col=0)
    test_sample = stratified_sample(test_df, esrd_ids, rng, "test")
    test_sample.to_csv(TEST_DST)
    print(f"Wrote {TEST_DST}")


if __name__ == "__main__":
    main()
