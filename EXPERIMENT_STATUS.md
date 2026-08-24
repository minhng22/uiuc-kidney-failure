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
| 2 | Mini-experiment (rep99) | re-run 2026-08-23 16:1x CDT (sunlab-serv-01) — 12/17 passed, 5/17 failed, all `four_features` on the same known sksurv "censoring survival function is zero" AUC edge case (cox/ddh/hazard_transformer/rnnsurv/kfre); no new failures | [report](generated_data/rep99/mini_experiment_status_report.txt) (original run; re-run not re-written, same known failure mode) |
| 2.1 | Feature-importance analysis | re-run 2026-08-23 16:18 CDT (sunlab-serv-01) — 3 scenario reports, all clean | [four_features](generated_data/rep99/four_features_shap_analysis_report.txt), [eight_features](generated_data/rep99/eight_features_shap_analysis_report.txt), [twenty_features_heterogeneous](generated_data/rep99/twenty_features_heterogeneous_shap_analysis_report.txt) |
| 2.1 | Additional analyses: calibration + decision-curve analysis, text + charts (rep99 sanity check) | re-checked again 2026-08-23 ~16:3x CDT (sunlab-serv-01) — found + fixed one more latent bug: all-NaN predictions (an undertrained-model failure mode, not currently occurring on rep99 but previously unguarded) silently produced an empty calibration table and a silently-biased net-benefit curve, same "no error, just wrong" class as earlier bugs — now explicitly detected and logged, verified with a synthetic NaN test; re-ran clean, no behavior change on current rep99 data. Earlier: re-run surfaced Brier score silently `None` for ddh/hazard_transformer (fixed); metrics double-check found 3 more bugs (hazard_transformer/logistic_hazard/ddh were using approximated instead of native model output; a degenerate-prediction display bug) — see plan doc "Metrics double-check" / "Another bug-check pass"; competing-risk analysis considered and declined (not planned) | reports: [four_features](generated_data/rep99/four_features_clinical_validity_report.txt), [eight_features](generated_data/rep99/eight_features_clinical_validity_report.txt), [twenty_features_heterogeneous](generated_data/rep99/twenty_features_heterogeneous_clinical_validity_report.txt); charts: `<scenario>_calibration_plot.png` / `<scenario>_decision_curve_plot.png` per scenario, plus cross-model `c_index_comparison.png` / `brier_comparison.png` / `auc_comparison.png`, all in `generated_data/rep99/` |
| 3.0 | rep1 full run + analysis report — approval gate before 3.1 | **not done** — rep1 was started, killed by user request before finishing (partial: cox/dynamic_deephit/hazard_transformer partially run — see report); needs relaunch | see Background processes below |
| 3.1 | rep2-4 full runs (blocked on 3.0's approval gate) | **not done** — rep2/rep3/rep4 were started (2 sessions) before Stage 3 was split into 3.0/3.1; all stopped/killed by user request before finishing; needs relaunch once 3.0 is approved | see Background processes below |

## Background processes

### Stage 3.0/3.1 (full experiment runs, rep1 launched under 3.0 + rep2 under what is now 3.1) — owner: session on sunlab-serv-02.cs.illinois.edu

**Stopped by user request** (2026-08-23 ~14:3x CDT) — `kill -TERM` on both process
groups, confirmed clean (no remaining processes). Both were healthy/actively computing
at kill time (verified via `/proc/<pid>/stat` CPU-time deltas moments before), not
stalled — killed on request, not due to a hang. rep3, rep4, rep5 also not running
(rep3/rep4 stopped earlier by user request per the sunlab-serv-03 session below; rep5
never started).

| PID | Rep | Log | Status |
|---|---|---|---|
| ~~2870156~~ | 1 | [eval_all_rep1.log](pkgs/scripts/eval_all_rep1.log) | **killed by user request** — was in progress (hazard_transformer, 20h11m elapsed); dynamic_deephit lost eight_features/twenty_features_heterogeneous to a bug (fixed + verified on rep99, not yet backfilled here) |
| ~~2870177~~ | 2 | [eval_all_rep2.log](pkgs/scripts/eval_all_rep2.log) | **killed by user request** — was in progress (hazard_transformer, 18h31m elapsed); same dynamic_deephit issue as rep1 |

Launched via `bash pkgs/scripts/run_rep.sh <rep>` (EXPERIMENTS: cox, dynamic_deephit,
hazard_transformer, logistic_hazard, rnnsurv, kfre). Verified the `CKD_FIFTY_FEATURES_HETEROGENEOUS`
guard (added during Stage 2) correctly skips that scenario for both reps.

**Bug found + fixed (2026-08-22/23), verifying on rep99 per the new rule "When a bug is found in
experiment code, verify the fix on rep99 first" (CLAUDE.md):** `dynamic_deephit` failed on
`four_features` for every currently-running rep (rep1/2 here, rep3/4 per sunlab-serv-03 below),
which — since its `__main__` had no per-scenario exception handling — silently skipped
`eight_features`/`twenty_features_heterogeneous` entirely for that model, for every rep. Two
distinct root causes found:
  1. **NaN loss**: `combine_loss()` in `pkgs/experiments/utils.py` called `torch.log()` directly
     on `sigmoid()`-activated hazard predictions with no epsilon clamp — float32 sigmoid
     saturation to exactly 0.0/1.0 (plausible given Optuna's learning_rate up to 1e-2, no
     gradient clipping, and `uacr`'s heavy right skew) produces `log(0) = -inf`, poisoning the
     loss permanently. Fixed: clamp hazard predictions to `[1e-7, 1-1e-7]` before `log()`.
  2. **AUC time-range error**: `dynamic_deephit.py`'s `auc()` computed `cumulative_dynamic_auc`
     per 16-patient mini-batch against a fixed 730-day grid, instead of once over the whole test
     set like `cox.py` — a small batch has a much higher chance of a shorter max follow-up than
     the full test set, triggering sksurv's hard validation error. Fixed: accumulate across all
     batches and compute once, with `times` bounded to the test set's actual observed follow-up.
  3. Also added per-scenario try/except in `dynamic_deephit.py`'s `__main__` so one scenario's
     failure no longer skips the rest.
Rep99 verification in progress: PID 3071682, `CKD_REP=99 PYTHONPATH=. python -m pkgs.experiments.dynamic_deephit`,
log [rep99_verify_ddh_fix_20260823_132640.log](pkgs/scripts/logs/rep99_verify_ddh_fix_20260823_132640.log).
Stale rep99 `*_ddh_model.pt` cleared first so this is a clean re-train under the fixed loss.

Rep99 verification progress (13:49 CDT): four_features done — 10/10 trials
clean (no NaN, best C-index 0.864), hit the known (unrelated) censoring-edge-
case on AUC, isolation caught it and continued. eight_features done — 10/10
trials clean, no NaN, no errors. Now on twenty_features_heterogeneous (the
scenario that ORIGINALLY failed with NaN at rep99) — trials producing real
values so far (0.57-0.77), no NaN yet. Still in progress.

**Rep99 verification: PASSED.** four_features (10/10 trials clean, hit only
the known unrelated censoring-edge-case, caught by isolation); eight_features
(fully clean, AUC 0.82/Brier 0.247); twenty_features_heterogeneous (the
scenario that ORIGINALLY failed with NaN at rep99 — now trains/evaluates
cleanly, C-index 0.442/AUC 0.54, Brier not computable due to a benign
small-sample time-range warning, not a crash). No NaN anywhere. Both fixes
+ the per-scenario isolation confirmed working.

Now re-running Stage 2.1 feature-importance analysis on rep99 (stale
reports/plots cleared first, since dynamic_deephit models changed): PID
3080173, log [stage21_feature_importance_verify_20260823_140224.log](pkgs/scripts/logs/stage21_feature_importance_verify_20260823_140224.log).
2 of 3 done (four_features, eight_features), now on twenty_features_heterogeneous.

Last Updated: 2026-08-23 14:15 CDT (sunlab-serv-02.cs.illinois.edu)

### Stage 3.1 (full experiment runs, rep3/rep4) — owner: session on sunlab-serv-03.cs.illinois.edu

**Running: rep3, rep4** (next ≤2 not-yet-running/done reps, per EXPERIMENT_PLAN_DETAILS.md Stage 3's
per-session cap; rep1/rep2 already running under the sunlab-serv-02 session above). rep5 left for a
later "run stage 3" / "run the rest".

| PID | Rep | Log | Status |
|---|---|---|---|
| — | 3 | [eval_all_rep3.log](pkgs/scripts/eval_all_rep3.log) | **stopped by user request** — was mid-fix-relaunch after the hazard_transformer stall (see note); not currently running |
| — | 4 | [eval_all_rep4.log](pkgs/scripts/eval_all_rep4.log) | **stopped by user request** — killed for the same hazard_transformer stall; relaunch was rejected/never started; not currently running |

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

UPDATE 2026-08-23 13:1x CDT: both rep3 and rep4's `hazard_transformer` were found stuck for
16+ hours on `twenty_features_heterogeneous` (0% GPU utilization, zero I/O, no log progress —
see prior note below this one once written... see full diagnosis inline above this line).
Root cause: `HazardTransformerDataset.__getitem__` (and the equivalent classes in
`dynamic_deephit.py` and `logistic_hazard.py`) recomputed `self.df[col].mean()`/`.std()` over
the FULL dataframe on every single subject access — O(subjects x features x N), invisible at
rep99's tiny mini-experiment scale but catastrophic at full Stage 3 scale (6.5M rows x 20
columns x 26k subjects). Fixed by caching each column's mean/std once per Dataset instance in
all three files ([hazard_transformer.py](pkgs/experiments/hazard_transformer.py),
[dynamic_deephit.py](pkgs/experiments/dynamic_deephit.py),
[logistic_hazard.py](pkgs/experiments/logistic_hazard.py)) — verified numerically identical
output vs. the old inline computation, and benchmarked the real fix: full-epoch `__getitem__`
cost on rep3's actual `twenty_features_heterogeneous` train data (6,512,638 rows / 26,080
subjects) dropped from "never completes" to ~368s estimated.
Killed both stuck processes (`kill -TERM` on their process groups) and relaunched rep3
(new PID 2460715). Before relaunching rep4, **the user asked to terminate the running rep(s)**
— rep3 (PID 2460715) was killed immediately after starting; rep4's relaunch had already been
blocked/rejected and never started. **Both rep3 and rep4 are currently NOT RUNNING** — stopped
per user request, pending the user's direction on whether/how to resume. The
`dynamic_deephit`/`four_features` failures noted below (both reps) are unrelated to this stall
and still apply to whatever partial run state exists on disk.

Last Updated: 2026-08-23 13:20 CDT (sunlab-serv-03.cs.illinois.edu)
