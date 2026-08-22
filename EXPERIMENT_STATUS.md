# Feature Set Experiment Plan (4/8/20 features)

Tracks execution of [EXPERIMENT_PLAN_DETAILS.md](EXPERIMENT_PLAN_DETAILS.md) (Stage 0 plan,
approved). Do not restart another session's row without confirming its host is actually dead.

## Status

| Stage | Task | Status | Notes |
|---|---|---|---|
| 1a | Task A: determine 20 lab features | **done** | Report: [generated_data/rep1/twenty_features_lab_analysis_report.txt](generated_data/rep1/twenty_features_lab_analysis_report.txt). |
| 1a | Task B: locate Tangri et al. 8-variable KFRE coefficients | **done, fully primary-sourced, no caveats** | Report: [generated_data/rep1/kfre_8variable_coefficients_report.txt](generated_data/rep1/kfre_8variable_coefficients_report.txt). |
| — | **Stage 1a complete, zero open caveats — awaiting user approval before 1b/1c per EXPERIMENT_PLAN_DETAILS.md's explicit stop-and-wait gate.** | | |
| 1b | Code changes (commons.py, types.py, time_series_store.py, model_data_store.py, experiments/*.py) | **done except kfre.py** | types.py enum, commons.py path constants, store.py `get_uacr_df`, time_series_store.py merge logic + scenario branches, model_data_store.py `get_train_test_data` branches, and all 5 experiments/*.py (cox/ddh/hazard_transformer/logistic_hazard/rnnsurv) updated + unit-tested (merge_nearest_within_admission verified against synthetic data matching EXPERIMENT_PLAN_DETAILS.md's worked example). `kfre.py` (closed-form 4-/8-var KFRE) still pending — not needed until Stage 2, so not blocking 1c-0. |
| 1c-0 | Pilot extraction (rep1 only) + cohort-flow analysis, **approval gate** | **extraction done, awaiting approval** | see Background processes below |
| 1c | Data extraction (rep1-5, parallel) | not started | blocked on 1c-0 approval |
| 2 | Mini-experiment (rep99) | not started | blocked on 1c |
| 2.1 | Feature-importance analysis | not started | blocked on 2 |
| 3 | Full experiment runs (rep1-5) | not started | blocked on 2 |

## Background processes

| PID | Host | Launch command | Log | Start time | Status |
|---|---|---|---|---|---|
| ~~1027916/1027933~~ | sunlab-serv-01.cs.illinois.edu | superseded run (same-admission uACR window) | [...extraction_20260821_195128.log](pkgs/scripts/logs/pilot_rep1_new_scenarios_extraction_20260821_195128.log) | 2026-08-21 19:51 CDT | **killed** — `four_features` finished under the old design and showed severe, outcome-correlated attrition (see report); killed before `eight_features`/`twenty_features_heterogeneous` completed so the fix below could apply to all three. Stale `four_features_{train,test}_data.csv` renamed to `*.old_same_admission_uacr_bak_20260821_2012.csv`, not deleted. |
| ~~1032596~~ | sunlab-serv-01.cs.illinois.edu | `CKD_REP=1 python -c "..."` ran `get_train_test_data` for `FOUR_FEATURES`, `EIGHT_FEATURES`, `TWENTY_FEATURES_HETEROGENEOUS` in sequence, **with uACR matching loosened to whole-patient-history** (`by='subject_id'` instead of `by='hadm_id'`, per user direction — see EXPERIMENT_PLAN_DETAILS.md "1a-2" addendum) | [...extraction_v2_20260821_201233.log](pkgs/scripts/logs/pilot_rep1_new_scenarios_extraction_v2_20260821_201233.log) | 2026-08-21 20:12 CDT | **finished cleanly** (all 3 scenarios, ~2h1m total: four_features 12.3min, eight_features 34.6min, twenty_features_heterogeneous 120.4min). Process exited, nothing still running. |

Deliberately did **not** run `model_data_store.py`'s full `__main__` (which now also includes
`CKD_FIFTY_FEATURES_HETEROGENEOUS`) because `generated_data/rep1/ckd_fifty_features_heterogeneous_{train,test}_data.csv`
are currently missing/renamed to `*.old_schema_bak_20260820_192307.csv` (another session's in-progress
schema migration, dated yesterday) — invoking the full `__main__` would have auto-regenerated that
scenario's data too, an expensive, unrelated, and possibly-conflicting side effect. Scoped this pilot to
just the 3 new scenarios via a direct `get_train_test_data` call per scenario instead.

### 1c-0 status: all 3 scenarios extracted, awaiting approval to proceed to 1c

Same-admission uACR bound (first pilot pass) caused severe, outcome-correlated attrition (34,332 →
1,220 patients for `four_features`, 98.2% positive vs. source's 91.9%) — killed, loosened per user
direction to whole-patient-history matching (row above), reran clean. Final results, patient-level:

| Scenario | N patients (% of source) | N records | Outcome-positive rate (source: 91.87%) |
|---|---|---|---|
| `four_features` | 2,809 (8.18%) | 26,829 | 83.52% (−8.36 pts) |
| `eight_features` | 1,213 (3.53%) | 5,407 | 85.08% (−6.80 pts) |
| `twenty_features_heterogeneous` | 32,601 (94.96%) | 8,126,090 | 91.45% (−0.42 pts) |

Reports (N/records/outcome rate — the current minimal format; an expanded Table-1-style version is
proposed in EXPERIMENT_PLAN_DETAILS.md "1c-0" but not yet implemented, pending approval):
[four_features](generated_data/rep1/four_features_cohort_flow_report.txt),
[eight_features](generated_data/rep1/eight_features_cohort_flow_report.txt),
[twenty_features_heterogeneous](generated_data/rep1/twenty_features_heterogeneous_cohort_flow_report.txt).

Two bugs caught and fixed while building these: (1) the report script's "original cohort" baseline used
the wrong function (`get_ckd_patients_and_diagnoses`, CKD-3-5-only, 10,179 patients) instead of the
pipeline's actual source population (CKD-3-5 OR ESRD dx, 34,332); (2) initial live commentary conflated
raw-merge-stage counts with final counts and row-level vs. patient-level outcome rate — corrected, and
the report now explicitly labels/caveats both.

Awaiting user sign-off on: (a) the expanded Table-1-style report content (EXPERIMENT_PLAN_DETAILS.md
"1c-0" analysis table), and (b) proceeding to full 1c (5-rep parallel extraction).

Last Updated: 2026-08-21 23:05 CDT (sunlab-serv-01.cs.illinois.edu)
