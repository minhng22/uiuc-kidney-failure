# Feature Set Experiment Plan (4/8/20 features)

Tracks execution of [EXPERIMENT_PLAN_DETAILS.md](EXPERIMENT_PLAN_DETAILS.md) (Stage 0 plan,
approved). Do not restart another session's row without confirming its host is actually dead.

## Status

| Stage | Task | Status | Notes |
|---|---|---|---|
| 1a | Task A: determine 20 lab features | **done** | Report: [generated_data/rep1/twenty_features_lab_analysis_report.txt](generated_data/rep1/twenty_features_lab_analysis_report.txt). |
| 1a | Task B: locate Tangri et al. 8-variable KFRE coefficients | **done, fully primary-sourced, no caveats** | Report: [generated_data/rep1/kfre_8variable_coefficients_report.txt](generated_data/rep1/kfre_8variable_coefficients_report.txt). |
| — | **Stage 1a complete, zero open caveats — awaiting user approval before 1b/1c per EXPERIMENT_PLAN_DETAILS.md's explicit stop-and-wait gate.** | | |
| 1b | Code changes (commons.py, types.py, time_series_store.py, model_data_store.py, experiments/*.py, kfre.py) | not started | blocked on 1a approval |
| 1c | Data extraction (rep1-5, parallel) | not started | blocked on 1b |
| 2 | Mini-experiment (rep99) | not started | blocked on 1c |
| 2.1 | Feature-importance analysis | not started | blocked on 2 |
| 3 | Full experiment runs (rep1-5) | not started | blocked on 2 |

## Background processes

None launched yet (Task A/B were synchronous, foreground, read-only — no PID to track until
Stage 1c's extraction runs are launched).

Last Updated: 2026-08-21 (sunlab-serv-01.cs.illinois.edu)
