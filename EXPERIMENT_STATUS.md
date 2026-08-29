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
| 2 | Mini-experiment (rep99) | done — 12/17 combos clean, 5 hit a known non-blocking AUC edge case | [report](generated_data/rep99/mini_experiment_status_report.txt) |
| 2.1 | Feature-importance + calibration/decision-curve analysis (rep99) | done | reports/charts in `generated_data/rep99/` |
| 2 | PMF-fix rerun (rep99) | done | [report](generated_data/rep99/stage2_pmf_fix_rerun_report.txt) |
| 2.1 | PMF-fix rerun analyses (rep99) | done | [report](generated_data/rep99/stage2_pmf_fix_rerun_report.txt) |
| 2.2 | Debug self-check on Stage 2.1's rep99 analysis | done — Findings #1-#3 fixed, verified, closed; 2 open questions remain | [report](generated_data/rep99/stage2_2_debug_report.txt) |
| 3.0 | rep1 four_features/eight_features run (all 11 models) + analysis | **retraining, 9/11 done** (picking up Findings #1-#3) | [report](generated_data/rep1/stage3_0_rep1_run_report.txt) |
| 3.0.1-3.0.3 | Debug → twenty_features_heterogeneous → Debug | queued to run automatically once 3.0 finishes (user-approved) | [plan](EXPERIMENT_PLAN_DETAILS.md) |
| 3.1 | rep2-4 full runs | not started, blocked on 3.0 | — |

## Background processes

| PID | Rep | Log | Status |
|---|---|---|---|
| 2940051 | 1 | [eval_all_rep1_stage3_0_four_eight.log](pkgs/scripts/eval_all_rep1_stage3_0_four_eight.log) | 9/11 done, no errors — see [report](generated_data/rep1/stage3_0_rep1_run_report.txt) for detail |

Last Updated: 2026-08-28 23:56 CDT (sunlab-serv-01.cs.illinois.edu). Full
history for this stage (launches, incidents, health checks) is in
[stage3_0_rep1_run_report.txt](generated_data/rep1/stage3_0_rep1_run_report.txt)
and [stage3_0_background_process_log.txt](generated_data/rep1/stage3_0_background_process_log.txt);
Stage 2.2's Findings #1-#3 are in
[stage2_2_debug_report.txt](generated_data/rep99/stage2_2_debug_report.txt).
