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
| 2 | Mini-experiment (rep99) | done — 17 runs, 11 passed, 6 failed (all explained) | [report](generated_data/rep99/mini_experiment_status_report.txt) |
| 2.1 | Feature-importance analysis | done — 3 scenario reports, all clean | [four_features](generated_data/rep99/four_features_shap_analysis_report.txt), [eight_features](generated_data/rep99/eight_features_shap_analysis_report.txt), [twenty_features_heterogeneous](generated_data/rep99/twenty_features_heterogeneous_shap_analysis_report.txt) |
| 3 | Full experiment runs (rep1-5) | **rep1, rep2, rep3, rep4 running** (max 2/session per plan, two sessions); rep5 not yet started | see Background processes below |

## Background processes

### Stage 3 (full experiment runs) — owner: session on sunlab-serv-02.cs.illinois.edu

**Running: rep1, rep2** (max 2 per session, per EXPERIMENT_PLAN_DETAILS.md Stage 3's cap).
rep3, rep4, rep5 not yet started by this session — see the sunlab-serv-03 row below for rep3/rep4.

| PID | Rep | Log | Status |
|---|---|---|---|
| 2870156 | 1 | [eval_all_rep1.log](pkgs/scripts/eval_all_rep1.log) | in progress (cox, on twenty_features_heterogeneous) |
| 2870177 | 2 | [eval_all_rep2.log](pkgs/scripts/eval_all_rep2.log) | in progress (cox, on twenty_features_heterogeneous) |

Launched via `bash pkgs/scripts/run_rep.sh <rep>` (EXPERIMENTS: cox, dynamic_deephit,
hazard_transformer, logistic_hazard, rnnsurv, kfre). Verified the `CKD_FIFTY_FEATURES_HETEROGENEOUS`
guard (added during Stage 2) correctly skips that scenario for both reps — confirmed via
`/proc/<pid>/fd` showing each cox process reading its own rep's `twenty_features_heterogeneous_train_data.csv`,
not `labevents.csv`.

Last Updated: 2026-08-22 17:16 CDT (sunlab-serv-02.cs.illinois.edu)

### Stage 3 (full experiment runs) — owner: session on sunlab-serv-03.cs.illinois.edu

**Running: rep3, rep4** (next ≤2 not-yet-running/done reps, per EXPERIMENT_PLAN_DETAILS.md Stage 3's
per-session cap; rep1/rep2 already running under the sunlab-serv-02 session above). rep5 left for a
later "run stage 3" / "run the rest".

| PID | Rep | Log | Status |
|---|---|---|---|
| 2204482 | 3 | [eval_all_rep3.log](pkgs/scripts/eval_all_rep3.log) | cox done; dynamic_deephit done (four_features FAILED — NEW NaN-loss issue, see note); in progress (hazard_transformer) |
| 2204775 | 4 | [eval_all_rep4.log](pkgs/scripts/eval_all_rep4.log) | cox done; dynamic_deephit done (four_features AUC step failed — known sksurv edge case, see note); in progress (hazard_transformer) |

Launched via `bash pkgs/scripts/run_rep.sh <rep>`. Before launching, found `eval_all_rep3.log`,
`eval_all_rep4.log`, `eval_all_rep5.log` (+ `run_rep3/5_master.log`, `run_rep{3,5}.pid`) already
present on disk, dated 2026-08-17 to 2026-08-22 — verified these are stale debris from an unrelated,
earlier `CKD_FIFTY_FEATURES_HETEROGENEOUS` experiment run (rep4's even used a separate isolated run
dir, `~/kidney-rep4-run`, not this repo's `run_rep.sh`), not this Stage 3 (four/eight/twenty-features)
run — no live process on this host held them, and rep3/rep4 already had the correct
`four_features`/`eight_features`/`twenty_features_heterogeneous` train/test CSVs from Stage 1c with no
`ckd_fifty_features_heterogeneous_train_data.csv` at the exact guarded path, so `cox.py`'s guard
correctly skips that scenario for both. Overwritten by this run as expected.

dynamic_deephit/rep4/four_features errored (`ValueError: all times must be within
follow-up time of test data`) after completing its 10-trial Optuna search — same known
sksurv `cumulative_dynamic_auc` edge case already documented in
[generated_data/rep99/mini_experiment_status_report.txt](generated_data/rep99/mini_experiment_status_report.txt)
(five occurrences there across cox/dynamic_deephit/hazard_transformer/logistic_hazard/rnnsurv,
all on `four_features` — too few censored patients at the later fixed AUC evaluation time
points relative to this scenario's smaller test-set follow-up range). Deterministic given the
data, not a bug to fix — `run_rep.sh` recorded it as a failed experiment and moved on to
`hazard_transformer` automatically, as designed; not relaunching.

dynamic_deephit/rep3/four_features also errored, but with a DIFFERENT root cause than
rep4's known sksurv AUC edge case: `ValueError: NaNs detected in inputs, please correct or
drop.` — Optuna trial 7 failed with `NaNs detected` mid-search (caught, search continued),
then the post-search final-model evaluation also produced all-NaN risk scores
(`Average Loss: nan` during training, early stopping, `hazards shape: [nan]`), crashing
`lifelines`' concordance_index and exiting the whole `dynamic_deephit` process (code 1).
**This is new information, not just a recurrence of a known issue**: Stage 2's
[mini_experiment_status_report.txt](generated_data/rep99/mini_experiment_status_report.txt)
saw this same NaN-loss pattern only on the tiny 20-patient `twenty_features_heterogeneous`
rep99 subsample and hypothesized it was small-sample-specific ("worth a closer look ... if
dynamic_deephit is expected to run on similarly-small future subsets"). Here it recurred on
the FULL-SCALE `four_features` data (2,247 train patients, rep3) — contradicting that
hypothesis; looks more like a hyperparameter-region training-instability issue in
`dynamic_deephit.py`, not a data-size artifact. Not fixing/relaunching now (out of scope
for a status check, and `run_rep.sh` already tolerated it and moved on to
`hazard_transformer` as designed, same as rep4) — flagging for the user to decide whether
this needs a closer look before trusting rep3/rep4's `dynamic_deephit`/`four_features`
results, and whether rep1/rep2/rep5 should be watched for the same.

Last Updated: 2026-08-22 21:11 CDT (sunlab-serv-03.cs.illinois.edu)
