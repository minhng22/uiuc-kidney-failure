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
| 2 | Mini-experiment (rep99) | **rerunning** — all rep99 models deleted and retrained from scratch (picks up DDH/HazardTransformer/RNN-Surv/DeepSurv fixes + the 5 newly-added models found/added this session) — see Background processes | [report](generated_data/rep99/mini_experiment_status_report.txt) |
| 2.1 | Feature-importance analysis + additional analyses (calibration + decision-curve, rep99 sanity check) | **queued** — will rerun once Stage 2's rep99 retrain finishes | SHAP reports: [four_features](generated_data/rep99/four_features_shap_analysis_report.txt), [eight_features](generated_data/rep99/eight_features_shap_analysis_report.txt), [twenty_features_heterogeneous](generated_data/rep99/twenty_features_heterogeneous_shap_analysis_report.txt); clinical-validity reports: [four_features](generated_data/rep99/four_features_clinical_validity_report.txt), [eight_features](generated_data/rep99/eight_features_clinical_validity_report.txt), [twenty_features_heterogeneous](generated_data/rep99/twenty_features_heterogeneous_clinical_validity_report.txt); charts: `<scenario>_calibration_plot.png` / `<scenario>_decision_curve_plot.png` per scenario, plus cross-model `c_index_comparison.png` / `brier_comparison.png` / `auc_comparison.png`, all in `generated_data/rep99/` |
| 3.0 | rep1 full run + analysis report — approval gate before 3.1. Model scope widened by user decision 2026-08-27: now 11 models (original 6 + deepsurv/gbsa/srf/survival_svm/weibul, all launched via `run_rep.sh` — their `__main__` blocks now cover the 3 new scenarios directly) | **stopped** (killed by user request 2026-08-27, not resumed; not yet relaunched with the widened model scope) — see [report](generated_data/rep1/stage3_0_rep1_run_report.txt) | see Background processes below |
| 3.1 | rep2-4 full runs (blocked on 3.0's approval gate) | **not done** — rep2/rep3/rep4 were started (2 sessions) before Stage 3 was split into 3.0/3.1; all stopped/killed by user request before finishing; needs relaunch once 3.0 is approved | see Background processes below |

## Background processes

### Stage 2 rerun (rep99) — owner: session on sunlab-serv-02.cs.illinois.edu

| PID | Rep | Log | Status |
|---|---|---|---|
| 3948360 | 99 | [stage2_rerun_rep99.log](pkgs/scripts/logs/stage2_rerun_rep99.log) | in progress — cox/ddh/hazard_transformer/logistic_hazard/rnnsurv/kfre, official Stage 2 driver (`pkgs/scripts/run_stage2_new_scenarios.py`) |
| 3948361 | 99 | [stage3_extra_models_rerun_rep99.log](pkgs/scripts/logs/stage3_extra_models_rerun_rep99.log) | in progress — deepsurv/gbsa/srf/survival_svm/weibul (added this session, not part of the official Stage 2 plan but rebuilt alongside it since their models were deleted too) |

### Stage 3.0 (rep1 full run) — owner: session on sunlab-serv-01.cs.illinois.edu

(none currently active — PID 2449903 and its subprocesses killed by user
request 2026-08-27; confirmed dead via `ps -p`, no python training processes
or GPU usage remain on this host; see
[report](generated_data/rep1/stage3_0_rep1_run_report.txt) for what was in
flight and needs relaunching)

Launch/reuse-plan details: [report](generated_data/rep1/stage3_0_rep1_run_report.txt).
`get_device()` GPU-selection change + parallel experiment launching + this resume
run: [report](generated_data/rep1/stage3_0_parallel_resume_report.txt).

Ad-hoc `ddh`/`four_features`+`eight_features` eval (user request, PID 2042631, separate from
the main rep1 run above) — done, see [report](generated_data/rep1/four_eight_features_ddh_eval_report.txt).

TEMPORARY/interim Stage 2.1-style analysis for rep1 (user request, everything trained so
far — not the final report) — done, see [check log](generated_data/rep1/stage3_0_background_process_log.txt)
for full checklist; reports/charts alongside the rep99 ones in `generated_data/rep1/`.

Last Updated: 2026-08-27 (sunlab-serv-01.cs.illinois.edu) — killed PID 2449903
(Stage 3.0 rep1 run, ddh/hazard_transformer/rnnsurv) by user request; confirmed
dead via `ps -p`, no remaining python/GPU activity on this host. Also: per user
decision, widened Stage 3.0/3.1 model scope to include deepsurv/gbsa/srf/
survival_svm/weibul for every rep (previously rep99-only). Rejected the
initial approach (a separate scoped driver script) per user feedback —
instead edited each of the 5 models' own `__main__` blocks (user-approved) to
also run `run_scenario()` for four_features/eight_features/
twenty_features_heterogeneous, matching the pattern already used by
cox/ddh/hazard_transformer/logistic_hazard/rnnsurv, and added all 5 to
`run_rep.sh`'s `EXPERIMENTS` array — no extra script needed. Updated
`EXPERIMENT_PLAN_DETAILS.md` Stage 3.0/3.1 sections accordingly, and removed
`STAGE3_EXTRA_MODELS_EXPERIMENT_PLAN.md` (superseded — its rep99 findings
live permanently in `generated_data/rep99/stage3_extra_models_report.txt`).