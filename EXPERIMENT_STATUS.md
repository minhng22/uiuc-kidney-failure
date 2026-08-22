# Feature Set Experiment Plan (4/8/20 features)

Tracks execution of [EXPERIMENT_PLAN_DETAILS.md](EXPERIMENT_PLAN_DETAILS.md) (Stage 0 plan,
approved). Do not restart another session's row without confirming its host is actually dead.

## Status

| Stage | Task | Status | Notes |
|---|---|---|---|
| 1a | Task A: determine 20 lab features | done | [report](generated_data/rep1/twenty_features_lab_analysis_report.txt) |
| 1a | Task B: locate Tangri et al. 8-variable KFRE coefficients | done | [report](generated_data/rep1/kfre_8variable_coefficients_report.txt) |
| 1b | Code changes (types/commons/time_series_store/model_data_store/experiments/kfre) | done | [report](generated_data/rep1/stage1b_implementation_report.txt) |
| 1c-0 | Pilot extraction (rep1) + cohort-flow analysis — approval gate | done, approved | [report](generated_data/rep1/stage1c0_pilot_extraction_report.txt) |
| 1c | Full extraction (rep2-5, parallel) | done | [report](generated_data/rep1/stage1c_full_extraction_report.txt) |
| 2 | Mini-experiment (rep99) | in progress | [report](generated_data/rep99/mini_experiment_status_report.txt) |
| 2.1 | Feature-importance analysis | not started | blocked on 2 |
| 3 | Full experiment runs (rep1-5) | not started | blocked on 2 |

## Background processes

Only currently-active processes are listed here. Full history (superseded
runs, incidents, fixes) lives in each stage's report, linked above and
below.

### Stage 1c (rep2-5 extraction) — owner: session on sunlab-serv-01.cs.illinois.edu

**Done.** rep5's retry (PID 1218942) finished cleanly (`ALL DONE rep5 (retry)`, 7371.6s for
`twenty_features_heterogeneous`). All 5 reps now have complete, verified `four_features`/
`eight_features`/`twenty_features_heterogeneous` train+test data (byte-identical file sizes across
reps, as expected — same source database and deterministic split). No processes running. This
session stops here per user instruction; does not extend into Stage 2/2.1/3.

Full history: [generated_data/rep1/stage1c_full_extraction_report.txt](generated_data/rep1/stage1c_full_extraction_report.txt).

Last Updated: 2026-08-22 (sunlab-serv-01.cs.illinois.edu)

### Stage 2 (rep99 mini-experiment) — owner: session on sunlab-serv-02.cs.illinois.edu

| PID | Log | Status |
|---|---|---|
| 2823675 | [log](pkgs/scripts/logs/rep99_stage2_scoped_20260822_131828.log) | in progress |

`twenty_features_heterogeneous`'s rep99 subsample was cut to 10 patients/class
(20 total, was 250/class) per user decision — see
[EXPERIMENT_PLAN_DETAILS.md addendum](EXPERIMENT_PLAN_DETAILS.md) and full
history: [generated_data/rep99/mini_experiment_status_report.txt](generated_data/rep99/mini_experiment_status_report.txt).

Last Updated: 2026-08-22 13:30 CDT (sunlab-serv-02.cs.illinois.edu) — resample
speedup confirmed (11min for cox+dynamic_deephit+2/3 hazard_transformer vs.
~45min/trial before); dynamic_deephit/twenty_features_heterogeneous now
fails with a new NaN-loss error at the smaller sample size, see report.
