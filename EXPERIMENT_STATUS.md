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
| 2 | Mini-experiment (rep99) | done | [report](generated_data/rep99/mini_experiment_status_report.txt) |
| 2.1 | Feature-importance analysis + additional analyses (calibration + decision-curve, rep99 sanity check) | done | SHAP reports: [four_features](generated_data/rep99/four_features_shap_analysis_report.txt), [eight_features](generated_data/rep99/eight_features_shap_analysis_report.txt), [twenty_features_heterogeneous](generated_data/rep99/twenty_features_heterogeneous_shap_analysis_report.txt); clinical-validity reports: [four_features](generated_data/rep99/four_features_clinical_validity_report.txt), [eight_features](generated_data/rep99/eight_features_clinical_validity_report.txt), [twenty_features_heterogeneous](generated_data/rep99/twenty_features_heterogeneous_clinical_validity_report.txt); charts: `<scenario>_calibration_plot.png` / `<scenario>_decision_curve_plot.png` per scenario, plus cross-model `c_index_comparison.png` / `brier_comparison.png` / `auc_comparison.png`, all in `generated_data/rep99/` |
| 3.0 | rep1 full run + analysis report — approval gate before 3.1 | **running** — see [report](generated_data/rep1/stage3_0_rep1_run_report.txt) | see Background processes below |
| 3.1 | rep2-4 full runs (blocked on 3.0's approval gate) | **not done** — rep2/rep3/rep4 were started (2 sessions) before Stage 3 was split into 3.0/3.1; all stopped/killed by user request before finishing; needs relaunch once 3.0 is approved | see Background processes below |

## Background processes

### Stage 3.0 (rep1 full run) — owner: session on sunlab-serv-01.cs.illinois.edu

| PID | Rep | Log | Status |
|---|---|---|---|
| 1613542 | 1 | [eval_all_rep1.log](pkgs/scripts/eval_all_rep1.log) | in progress (dynamic_deephit only) — see [check log](generated_data/rep1/stage3_0_background_process_log.txt) |
| 2291673 | 1 | [eval_all_rep1_resume_no_ddh.log](pkgs/scripts/eval_all_rep1_resume_no_ddh.log) | in progress (hazard_transformer/rnnsurv, parallel, ddh excluded; cox+kfre+logistic_hazard done) — see [check log](generated_data/rep1/stage3_0_background_process_log.txt) |

Launch/reuse-plan details: [report](generated_data/rep1/stage3_0_rep1_run_report.txt).
`get_device()` GPU-selection change + parallel experiment launching + this resume
run: [report](generated_data/rep1/stage3_0_parallel_resume_report.txt).

Ad-hoc `ddh`/`four_features`+`eight_features` eval (user request, PID 2042631, separate from
the main rep1 run above) — done, see [report](generated_data/rep1/four_eight_features_ddh_eval_report.txt).

TEMPORARY/interim Stage 2.1-style analysis for rep1 (user request, everything trained so
far — not the final report) — done, see [check log](generated_data/rep1/stage3_0_background_process_log.txt)
for full checklist; reports/charts alongside the rep99 ones in `generated_data/rep1/`.

Last Updated: 2026-08-26 18:42 CDT (sunlab-serv-01.cs.illinois.edu)