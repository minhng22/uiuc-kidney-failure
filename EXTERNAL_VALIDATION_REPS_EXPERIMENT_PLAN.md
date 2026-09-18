# External-Validation Reps Experiment Plan (rep1–rep5, three-way split)

**Last Updated:** 2026-09-18 04:30 (sunlab-serv-02) — build running, rep1 twenty_features ~complete

## Goal

Build 5 reps (`rep1`–`rep5`) under a new patient-level split rule that adds a
held-out external validation set, replacing the two-way rule used by every
existing rep.

| | Rule | Fractions of the patient pool |
|---|---|---|
| Legacy (`get_train_test_data`, `random_state=42`) | pool → train/test | 80% train / 20% test |
| **This plan** | pool → 80% development / 20% external validation; development → 4:1 train:test | **64% train / 16% test / 20% external validation** |

Splits are patient-level (no `subject_id` straddles two splits) and stratified
on the patient's ESRD label (`max(has_esrd)` over their rows).

**Each rep gets its own seed (`42 + rep`).** The legacy splitter hardcodes
`random_state=42`, so all previous reps shared one identical split — finding 1
of [hazard_transformer_metrics_audit_report.txt](generated_data/hazard_transformer_metrics_audit_report.txt).

## Source pool

Reconstructed as the union of `generated_data/rep100/<scenario>_train_data.csv`
+ `_test_data.csv` (rep100 is the rep formerly at rep1, renamed outside these
sessions). Those two are disjoint by `subject_id` and together are exactly what
extraction produced, so **no re-extraction from `labevents.csv` is needed**
(that would be ~2h per scenario).

| Scenario | Pool patients | Train (64%) | Test (16%) | External val (20%) |
|---|---|---|---|---|
| four_features | 2,809 | 1,797 | 450 | 562 |
| eight_features | 1,213 | 776 | 194 | 243 |
| twenty_features_heterogeneous | 32,601 | ~20,865 | ~5,216 | ~6,520 |

ESRD prevalence: 83.5% (four_features), 85.1% (eight_features) — held to within
0.15pp across all three splits by the stratification.

## Concrete scripts

- **Builder:** [pkgs/scripts/build_external_validation_reps.py](pkgs/scripts/build_external_validation_reps.py)
  - `python -m pkgs.scripts.build_external_validation_reps` — all 5 reps, all 3 scenarios
  - `--reps 1 2` / `--scenarios four_features` — scope it down
  - `--dry-run` — report split sizes without writing CSVs
  - Streams the source CSVs in 200k-row chunks and routes rows by `subject_id`,
    so `twenty_features_heterogeneous` (6.5M train rows, 1.24GB) never loads whole.
  - Output CSVs keep the source shape (leading unnamed row-index column), so
    every existing reader behaves identically.
  - Also carries `esrd_patient_ids.csv` over from rep100 and writes
    `generated_data/rep<N>/external_validation_split_report.txt`.
- **Loaders:** `pkgs/data_analysis/model_data_store.py`
  - `has_external_validation_data(scenario)` — gate for reps built with the old rule
  - `get_external_validation_data(scenario)`
  - `get_train_test_external_data(scenario)` → `(train, test, external)`, asserts no patient leakage
- **Paths:** `pkgs/commons.py` — `<scenario>_external_validation_data_path`

## Status

| Step | Status | Notes |
|---|---|---|
| Builder script + loaders + commons paths | done | dry-run verified: split fractions, stratified event rates, and cross-rep partition independence (~20% pairwise external-set overlap, as expected for independent 20% draws) |
| Build rep1–rep5 data (3 scenarios) | four_features + eight_features done & verified for all 5 reps; twenty_features_heterogeneous running (rep1 written, reps 2–5 pending) | per-rep [report](generated_data/rep1/external_validation_split_report.txt) |
| Training runs on rep1–rep5 | not started | not in scope of this plan; no runs launched |

## Background processes

| PID | Host | Command | Log | Started | Status | Blocking / shared state |
|---|---|---|---|---|---|---|
| 1830033 | sunlab-serv-02 | `python -u -m pkgs.scripts.build_external_validation_reps` | [log](pkgs/scripts/logs/build_external_validation_reps.log) | 2026-09-18 | running (checked 04:30, 103% CPU) — twenty_features_heterogeneous rep1 written, reps 2–5 pending | writes `generated_data/rep1`–`rep5` only |

Log is unbuffered (`python -u`), so no py-spy snapshot is needed to read live progress.

**Shared state:** writes only to `generated_data/rep1` … `generated_data/rep5`,
and reads `generated_data/rep100` read-only. Those five rep dirs did not exist
on this host at launch. Does **not** touch `current_rep`/`CKD_REP` state of any
other run.
