"""Build reps with a three-way patient-level split: train / test / external validation.

Split rule (differs from the legacy two-way rule in
`pkgs/data_analysis/model_data_store.py:get_train_test_data`, which is a plain
80/20 train/test split of unique `subject_id`s at a fixed `random_state=42`):

    original pool of patients
      -> 80% development set, 20% external validation
      -> within development: 4:1 train:test

i.e. 64% train / 16% test / 20% external validation of the original patient
pool. Splits are patient-level (a `subject_id`'s rows never straddle two
splits) and stratified on the patient's ESRD label.

The "original pool" is reconstructed as the union of a source rep's existing
`<scenario>_train_data.csv` + `<scenario>_test_data.csv`. Those two are
disjoint by `subject_id` and together are exactly the cohort that extraction
produced, so no re-extraction from `labevents.csv` is needed (that would be a
~2h job per scenario).

Unlike the legacy splitter, each destination rep gets its own seed
(`BASE_SEED + rep`), so the 5 reps are genuinely different partitions. The
existing reps all shared one split because `random_state=42` is hardcoded --
see `generated_data/hazard_transformer_metrics_audit_report.txt` finding 1.

Patient ESRD label: `max(has_esrd)` over that patient's rows. `has_esrd` is 1
only on the record(s) at the first ESRD date and later records are dropped
during extraction (`pkgs/data_analysis/time_series_store.py:104`), so the
per-patient max is the event indicator.

Output CSVs are written in the same shape as the source (leading unnamed
row-index column, contiguous 0..n-1 per file) so every existing reader
behaves identically.

Usage:
    python -m pkgs.scripts.build_external_validation_reps
    python -m pkgs.scripts.build_external_validation_reps --reps 1 2 --scenarios four_features
    python -m pkgs.scripts.build_external_validation_reps --dry-run
"""
import argparse
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pkgs.commons import project_dir
from pkgs.paths import repetition_directory_name

SOURCE_REP = 100
DEST_REPS = [1, 2, 3, 4, 5]
BASE_SEED = 42

EXTERNAL_VALIDATION_FRACTION = 0.2  # of the whole pool
TEST_FRACTION_OF_DEV = 0.2          # 4:1 train:test inside the development set

SCENARIOS = [
    "four_features",
    "eight_features",
    "twenty_features_heterogeneous",
]

# Files that live in a rep dir but are extraction *inputs* rather than outputs,
# and so must be carried over from the source rep (e.g. esrd_patient_ids_path).
CARRY_OVER_FILES = ["esrd_patient_ids.csv"]

CHUNK_SIZE = 200_000

SPLITS = ["train", "test", "external_validation"]


def rep_dir(rep: int) -> str:
    return f"{project_dir()}/generated_data/{repetition_directory_name(rep)}"


def source_paths(scenario: str) -> list:
    src = rep_dir(SOURCE_REP)
    return [
        f"{src}/{scenario}_train_data.csv",
        f"{src}/{scenario}_test_data.csv",
    ]


def dest_path(rep: int, scenario: str, split: str) -> str:
    suffix = {
        "train": "train_data",
        "test": "test_data",
        "external_validation": "external_validation_data",
    }[split]
    return f"{rep_dir(rep)}/{scenario}_{suffix}.csv"


def read_patient_labels(paths: list) -> pd.Series:
    """subject_id -> patient-level ESRD label (max of has_esrd), over the whole pool."""
    labels = {}
    for path in paths:
        for chunk in pd.read_csv(path, usecols=["subject_id", "has_esrd"], chunksize=CHUNK_SIZE):
            for subject_id, value in chunk.groupby("subject_id")["has_esrd"].max().items():
                labels[subject_id] = max(int(value), labels.get(subject_id, 0))
    series = pd.Series(labels, dtype=int).sort_index()
    series.index.name = "subject_id"
    return series


def split_patients(labels: pd.Series, seed: int) -> dict:
    """pool -> {80% dev -> 4:1 train/test}, {20% external validation}. Stratified on ESRD."""
    subjects = labels.index.to_numpy()
    y = labels.to_numpy()

    dev_subjects, ext_subjects, dev_y, _ = train_test_split(
        subjects, y,
        test_size=EXTERNAL_VALIDATION_FRACTION,
        random_state=seed,
        stratify=y,
    )
    train_subjects, test_subjects = train_test_split(
        dev_subjects,
        test_size=TEST_FRACTION_OF_DEV,
        random_state=seed,
        stratify=dev_y,
    )
    return {
        "train": set(train_subjects.tolist()),
        "test": set(test_subjects.tolist()),
        "external_validation": set(ext_subjects.tolist()),
    }


def write_splits(paths: list, assignment: dict, rep: int, scenario: str) -> dict:
    """Stream the source CSVs and route each row to its patient's split file.

    Rows keep their source order, and a patient's rows are contiguous in the
    source (a patient lives in exactly one source file), so per-subject
    ascending `duration_in_days` is preserved -- `get_last_observation_data`
    asserts on that.
    """
    handles = {}
    written = {split: 0 for split in SPLITS}
    header_done = {split: False for split in SPLITS}
    try:
        for split in SPLITS:
            handles[split] = open(dest_path(rep, scenario, split), "w", newline="")

        for path in paths:
            for chunk in pd.read_csv(path, index_col=0, chunksize=CHUNK_SIZE):
                for split in SPLITS:
                    part = chunk[chunk["subject_id"].isin(assignment[split])]
                    if part.empty:
                        continue
                    part = part.copy()
                    part.index = pd.RangeIndex(written[split], written[split] + len(part))
                    part.to_csv(handles[split], header=not header_done[split])
                    header_done[split] = True
                    written[split] += len(part)
    finally:
        for handle in handles.values():
            handle.close()

    return written


def build_rep(rep: int, scenario: str, labels: pd.Series, paths: list, dry_run: bool) -> str:
    seed = BASE_SEED + rep
    assignment = split_patients(labels, seed)

    lines = [f"--- rep{rep} / {scenario} (seed={seed}) ---"]
    pool_n = len(labels)
    for split in SPLITS:
        subjects = assignment[split]
        n = len(subjects)
        event_rate = labels.loc[sorted(subjects)].mean()
        lines.append(
            f"  {split:<20} patients {n:>7,} ({n / pool_n:6.2%} of pool)  "
            f"ESRD patients {event_rate:6.2%}"
        )

    overlap = (
        assignment["train"] & assignment["test"]
        | assignment["train"] & assignment["external_validation"]
        | assignment["test"] & assignment["external_validation"]
    )
    assert not overlap, f"patient overlap between splits: {sorted(overlap)[:10]}"
    covered = assignment["train"] | assignment["test"] | assignment["external_validation"]
    assert covered == set(labels.index.tolist()), "splits do not cover the pool exactly"

    if dry_run:
        lines.append("  (dry run -- no files written)")
        return "\n".join(lines)

    os.makedirs(rep_dir(rep), exist_ok=True)
    written = write_splits(paths, assignment, rep, scenario)
    for split in SPLITS:
        lines.append(f"  {split:<20} rows {written[split]:>10,}  -> {dest_path(rep, scenario, split)}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, nargs="+", default=DEST_REPS)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--dry-run", action="store_true",
                        help="report the split sizes without writing any CSVs")
    args = parser.parse_args()

    print(f"Source rep: rep{SOURCE_REP}")
    print(f"Destination reps: {args.reps}")
    print(f"Scenarios: {args.scenarios}")
    print(f"Rule: {1 - EXTERNAL_VALIDATION_FRACTION:.0%} development / "
          f"{EXTERNAL_VALIDATION_FRACTION:.0%} external validation; "
          f"within development {(1 - TEST_FRACTION_OF_DEV) / TEST_FRACTION_OF_DEV:.0f}:1 train:test "
          f"(= {(1 - EXTERNAL_VALIDATION_FRACTION) * (1 - TEST_FRACTION_OF_DEV):.0%} / "
          f"{(1 - EXTERNAL_VALIDATION_FRACTION) * TEST_FRACTION_OF_DEV:.0%} / "
          f"{EXTERNAL_VALIDATION_FRACTION:.0%} of the pool)")
    print(f"Stratified on patient-level ESRD label; seed = {BASE_SEED} + rep", flush=True)

    reports = {}
    for scenario in args.scenarios:
        paths = source_paths(scenario)
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print(f"\n[{scenario}] SKIPPING -- source data not found: {missing}", flush=True)
            continue

        print(f"\n=== {scenario} ===", flush=True)
        print(f"Reading patient labels from {paths}", flush=True)
        labels = read_patient_labels(paths)
        print(f"Pool: {len(labels):,} patients, {labels.mean():.2%} with ESRD", flush=True)

        for rep in args.reps:
            started = datetime.now()
            report = build_rep(rep, scenario, labels, paths, args.dry_run)
            print(report, flush=True)
            print(f"  elapsed {datetime.now() - started}", flush=True)
            reports.setdefault(rep, []).append(report)

    if args.dry_run:
        return 0

    for rep in args.reps:
        if rep not in reports:
            continue
        os.makedirs(rep_dir(rep), exist_ok=True)
        for name in CARRY_OVER_FILES:
            src = f"{rep_dir(SOURCE_REP)}/{name}"
            dst = f"{rep_dir(rep)}/{name}"
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"rep{rep}: carried over {name}", flush=True)

        report_path = f"{rep_dir(rep)}/external_validation_split_report.txt"
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"THREE-WAY SPLIT (train / test / external validation) -- rep{rep}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"Host:      {os.uname().nodename}\n")
            f.write(f"Built by:  python -m pkgs.scripts.build_external_validation_reps --reps {rep}\n")
            f.write(f"Source:    generated_data/rep{SOURCE_REP}/<scenario>_{{train,test}}_data.csv (union = original pool)\n")
            f.write(f"Seed:      {BASE_SEED} + {rep} = {BASE_SEED + rep}\n\n")
            f.write("Rule: pool -> 80% development / 20% external validation; development -> 4:1 train:test\n")
            f.write("      => 64% train / 16% test / 20% external validation of the patient pool.\n")
            f.write("      Patient-level (no subject_id straddles splits), stratified on max(has_esrd) per patient.\n\n")
            f.write("\n\n".join(reports[rep]) + "\n")
        print(f"rep{rep}: wrote {report_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
