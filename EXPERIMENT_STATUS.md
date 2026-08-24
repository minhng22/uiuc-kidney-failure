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
| 2.1 | Feature-importance analysis | done | [four_features](generated_data/rep99/four_features_shap_analysis_report.txt), [eight_features](generated_data/rep99/eight_features_shap_analysis_report.txt), [twenty_features_heterogeneous](generated_data/rep99/twenty_features_heterogeneous_shap_analysis_report.txt) |
| 2.1 | Additional analyses: calibration + decision-curve analysis, text + charts (rep99 sanity check) | done | reports: [four_features](generated_data/rep99/four_features_clinical_validity_report.txt), [eight_features](generated_data/rep99/eight_features_clinical_validity_report.txt), [twenty_features_heterogeneous](generated_data/rep99/twenty_features_heterogeneous_clinical_validity_report.txt); charts: `<scenario>_calibration_plot.png` / `<scenario>_decision_curve_plot.png` per scenario, plus cross-model `c_index_comparison.png` / `brier_comparison.png` / `auc_comparison.png`, all in `generated_data/rep99/` |
| 3.0 | rep1 full run + analysis report — approval gate before 3.1 | **running** — relaunched 2026-08-23 19:28 CDT (sunlab-serv-01), PID 1613542; reuses already-trained cox (all 3 scenarios) + ddh/four_features + hazard_transformer/four_features+eight_features from the earlier killed attempt, trains the rest fresh (ddh/eight+twenty, hazard_transformer/twenty, logistic_hazard/rnnsurv/kfre all 3) | see Background processes below |
| 3.1 | rep2-4 full runs (blocked on 3.0's approval gate) | **not done** — rep2/rep3/rep4 were started (2 sessions) before Stage 3 was split into 3.0/3.1; all stopped/killed by user request before finishing; needs relaunch once 3.0 is approved | see Background processes below |

## Background processes

### Stage 3.0 (rep1 full run) — owner: session on sunlab-serv-01.cs.illinois.edu

**Running: rep1.** Relaunched after confirming the prior sunlab-serv-02 attempt was
cleanly killed (their own note below confirms "no remaining processes"; also verified
locally — `ps -p 2870156` (their old PID) found nothing on this host either).

| PID | Rep | Log | Status |
|---|---|---|---|
| 1613542 | 1 | [eval_all_rep1.log](pkgs/scripts/eval_all_rep1.log) | started 2026-08-23 19:28 CDT, in progress |

Launched via `bash pkgs/scripts/run_rep.sh 1` (per EXPERIMENT_PLAN_DETAILS.md Stage 3.0 —
same script, no new one). Verified before launching: `run_rep.sh` calls each experiment
module's own `__main__`, which guards `CKD_FIFTY_FEATURES_HETEROGENEOUS` behind an
`os.path.exists` check (per CLAUDE.md's entry-point rule) — confirmed rep1 has no
`ckd_fifty_features_heterogeneous_train_data.csv`, so that scenario is skipped, not
silently triggering a full raw extraction.

Reuses the prior killed attempt's already-written model files rather than retraining
from scratch (each experiment's own `run()` loads if the model file already exists):
cox (all 3 new scenarios already trained), `ddh`/`four_features`,
`hazard_transformer`/`four_features`+`eight_features`. Trains fresh: `ddh`/`eight_features`
+`twenty_features_heterogeneous` (previously lost to the now-fixed NaN-loss bug),
`hazard_transformer`/`twenty_features_heterogeneous` (previously the scenario stuck 16+
hours on the now-fixed mean/std-caching bug), and `logistic_hazard`/`rnnsurv`/`kfre` for
all 3 scenarios (never started before being killed).

Per CLAUDE.md's 10-minute auto-check rule, will re-verify (`ps -p 1613542`, log tail)
and update this section every ~10 minutes until rep1 finishes or fails-and-is-relaunched.
Once done: produce rep1's Stage 2.1-style analysis report (SHAP + clinical-validity,
pointed at `CKD_REP=1`) per the plan doc, then stop for the Stage 3.0→3.1 approval gate.

Last Updated: 2026-08-23 19:29 CDT (sunlab-serv-01.cs.illinois.edu)