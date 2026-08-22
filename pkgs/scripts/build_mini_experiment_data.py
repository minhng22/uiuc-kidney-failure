"""
One-off script for building mini-experiment (rep99) train/test data.

Builds a small, stratified (by ESRD status) random subsample of each
scenario's rep1 train/test data, and writes it to an isolated rep99 slot so
it can be used with `CKD_REP=99` without touching rep1-5's live data or
model files.

Generalized (2026-08-21, EXPERIMENT_PLAN_DETAILS.md Stage 2) to loop over
all scenarios instead of hardcoding a single one. A scenario is skipped
(with a warning, not an error) if its rep1 source files aren't present —
e.g. `ckd_fifty_features_heterogeneous`'s rep1 data is currently mid
schema-migration by another session; this script must not touch it.

See CKD_FIFTY_FEATURES_mini_experiment_plan.md / EXPERIMENT_PLAN_DETAILS.md
for the full plan.
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

ESRD_IDS_SRC = f"{SRC_DIR}/esrd_patient_ids.csv"

# Scenario name -> train/test data file basename prefix.
SCENARIOS = [
    "ckd_fifty_features_heterogeneous",
    "four_features",
    "eight_features",
    "twenty_features_heterogeneous",
]


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


def build_scenario(scenario: str, esrd_ids: np.ndarray, rng: np.random.Generator) -> None:
    train_src = f"{SRC_DIR}/{scenario}_train_data.csv"
    test_src = f"{SRC_DIR}/{scenario}_test_data.csv"
    train_dst = f"{DST_DIR}/{scenario}_train_data.csv"
    test_dst = f"{DST_DIR}/{scenario}_test_data.csv"

    if not (os.path.exists(train_src) and os.path.exists(test_src)):
        print(f"[{scenario}] skipping: source data not found at {train_src} / {test_src}")
        return

    print(f"=== {scenario} ===")
    print(f"Reading train data from {train_src}")
    train_df = pd.read_csv(train_src, index_col=0)
    train_sample = stratified_sample(train_df, esrd_ids, rng, f"{scenario}/train")
    train_sample.to_csv(train_dst)
    print(f"Wrote {train_dst}")
    del train_df, train_sample

    print(f"Reading test data from {test_src}")
    test_df = pd.read_csv(test_src, index_col=0)
    test_sample = stratified_sample(test_df, esrd_ids, rng, f"{scenario}/test")
    test_sample.to_csv(test_dst)
    print(f"Wrote {test_dst}")
    del test_df, test_sample


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    esrd_ids = pd.read_csv(ESRD_IDS_SRC)["subject_id"].unique()

    for scenario in SCENARIOS:
        # Fresh RNG per scenario (seeded) so each scenario's sample is
        # reproducible independent of loop order / which scenarios are
        # skipped.
        rng = np.random.default_rng(RANDOM_SEED)
        build_scenario(scenario, esrd_ids, rng)


if __name__ == "__main__":
    main()
