# CKD Fifty Features — Mini Experiment Plan

**Status:** APPROVED and EXECUTING (launched 2026-08-19 22:26:44 CDT).
**Owner/host:** this session, `sunlab-serv-02.cs.illinois.edu`.

## Goal
Quick smoke-test of the full 5-model pipeline (cox, dynamic_deephit,
hazard_transformer, logistic_hazard, rnnsurv) on the `CKD_FIFTY_FEATURES_HETEROGENEOUS`
scenario, using a small random patient subsample instead of the full dataset, so it
finishes in a reasonable time instead of the multi-day runtimes seen on rep1-5.

Results go in **this file** (not `CKD_FIFTY_FEATURES_EXPERIMENT_PLAN.md`), per user
request.

## Data source (where the subsample comes from)
Both mini train and test sets are subsampled **from rep1 only**:
- Train source: `generated_data/rep1/ckd_fifty_features_heterogeneous_train_data.csv` (8.1M rows, ~26,277 patients)
- Test source: `generated_data/rep1/ckd_fifty_features_heterogeneous_test_data.csv` (2.1M rows, ~6,570 patients)
- ESRD labels source (for stratified sampling): `generated_data/rep1/esrd_patient_ids.csv`

rep1 was chosen because it's the file the user pointed to
(`generated_data/rep1/ckd_fifty_features_heterogeneous_test_data.csv`) and because
rep1-5's `CKD_FIFTY_FEATURES_HETEROGENEOUS` data files are confirmed byte-identical
across all 5 reps (per `CKD_FIFTY_FEATURES_EXPERIMENT_PLAN.md`'s md5sum check), so
rep1 is representative of any rep. Output goes to the new `generated_data/rep99/`
slot (read from rep1, write to rep99 — rep1's own files are never modified).

## Decisions already confirmed with user
1. **Isolated slot, not rep1**: rep1-5 currently have live background training for
   this exact scenario running on other hosts (per
   [CKD_FIFTY_FEATURES_EXPERIMENT_PLAN.md](CKD_FIFTY_FEATURES_EXPERIMENT_PLAN.md)).
   To avoid clobbering that in-progress work, the mini run uses a new, unused rep
   slot: **`CKD_REP=99` → `generated_data/rep99/`** (commons.py builds the path as
   `generated_data/rep{CKD_REP}`, so rep99 is the simplest way to get full isolation
   without editing shared `commons.py` logic).
2. **Subsample both train and test**, not just test (full train is 8.1M rows /
   ~26k patients; cox alone took ~2 days at full size on rep1).
3. **500 patients** each for the mini train/test batches, stratified by ESRD
   status (250 ESRD + 250 non-ESRD) reusing the same stratification convention
   already used by `pkgs/data_analysis/model_data_store.py`'s `sample()` helper,
   so both classes are guaranteed present.
4. **Apply the drafted `combine_loss` vectorized fix** in
   [pkgs/experiments/utils.py](pkgs/experiments/utils.py#L206) repo-wide before
   running, since the current nested-loop version is confirmed catastrophically
   slow (this fix was previously drafted/verified but not applied, per the main
   experiment plan doc).

## Steps (all completed)
1. **Applied the `combine_loss` fix** in
   [pkgs/experiments/utils.py](pkgs/experiments/utils.py#L206) (vectorized,
   cumsum/gather based). Numerically verified against the original nested-loop
   version across 5 random trials — matched to <1e-6 relative error. This is a
   repo-wide bug fix, so it also benefits the live rep1-5 runs going forward
   (does not touch already-running processes' already-loaded code).
2. **Built mini data** via new script
   [pkgs/scripts/build_mini_experiment_data.py](pkgs/scripts/build_mini_experiment_data.py):
   - Read rep1's `ckd_fifty_features_heterogeneous_train_data.csv` (26,277
     patients, 8.1M rows) and `..._test_data.csv` (6,570 patients, 2.1M rows).
   - Stratified-sampled 250 ESRD + 250 non-ESRD unique `subject_id`s
     independently for train and test, using `generated_data/rep1/esrd_patient_ids.csv`
     for labels (fixed random seed **42**, `numpy.random.default_rng`).
   - Result: mini train = 500 patients / 186,731 rows; mini test = 500 patients
     / 200,279 rows.
   - Wrote to `generated_data/rep99/ckd_fifty_features_heterogeneous_train_data.csv`
     and `..._test_data.csv` (rep1's own files were only read, never modified).
3. **Launched all 5 models in background**:
   ```bash
   conda activate minhn2   # env with torch/pandas etc.
   bash pkgs/scripts/run_rep.sh 99
   ```
   PID **2181834** (wrapper, `setsid`-detached), host
   `sunlab-serv-02.cs.illinois.edu`, started **2026-08-19 22:26:44 CDT**.
   Runs cox → dynamic_deephit → hazard_transformer → logistic_hazard → rnnsurv
   sequentially. Logs:
   - Rep log: `pkgs/scripts/eval_all_rep99.log`
   - Master log: `pkgs/scripts/run_rep99_master.log`
   - PID file: `pkgs/scripts/run_rep99.pid`
4. **Monitoring**: results/progress are recorded in the table below as stages
   complete, in this doc only (not the shared experiment plan doc).

## What will NOT be touched
- `generated_data/rep1/` through `rep5/` (data or model files) — read-only access
  to rep1's CSVs to build the sample only.
- `pkgs/commons.py` — no edits, no `current_rep` mutation (isolation achieved via
  `CKD_REP` env var only, same as rep2/rep3's existing standalone runs).
- Any currently-running background process for rep1-5.

## Output Artifacts
| Artifact | Path |
|----------|------|
| Mini train data | `generated_data/rep99/ckd_fifty_features_heterogeneous_train_data.csv` |
| Mini test data | `generated_data/rep99/ckd_fifty_features_heterogeneous_test_data.csv` |
| Cox model | `generated_data/rep99/ckd_fifty_features_heterogeneous_cox_model.dill` |
| DDH model | `generated_data/rep99/ckd_fifty_features_heterogeneous_ddh_model.pt` |
| Hazard Transformer model | `generated_data/rep99/ckd_fifty_features_heterogeneous_hazard_transformer_model.pt` |
| Logistic Hazard model | `generated_data/rep99/ckd_fifty_features_heterogeneous_logistic_hazard_model.pt` |
| RNN Surv model | `generated_data/rep99/ckd_fifty_features_heterogeneous_rnn_surv_model.pt` |
| Run log | `pkgs/scripts/eval_all_rep99.log` |
| Master log | `pkgs/scripts/run_rep99_master.log` |
| PID file | `pkgs/scripts/run_rep99.pid` |

## Progress Tracking (filled in after launch)

| Model | Status | PID/Host | Start | End | C-index | Brier | AUC | Notes |
|-------|--------|----------|-------|-----|---------|-------|-----|-------|
| cox | ✅ done | wrapper PID 2181834, cox process PID 2181871 / sunlab-serv-02 | 22:26:44 CDT | 22:28:07 CDT (Aug 19) | 0.461 | 0.644 | 0.4641 | Completed in **83s** (vs. ~2 days at full rep1 scale) — confirms rep99 isolation + small sample works as intended. |
| dynamic_deephit | ❌ terminated by user | PID 2182603 (wrapper 2181834) / sunlab-serv-02 | 22:28:07 CDT (Aug 19) | terminated 2026-08-20 (user request) | - | - | - | Terminated via `kill -TERM -2181834` (whole process group) after ~19h with 0 Optuna trials completed and no GPU usage (see root-cause note above, `get_device()` hardcoded to `"cpu"`). Confirmed both the wrapper and the `dynamic_deephit` process are gone; no orphaned processes remain. `run_rep99.pid` removed. |
| hazard_transformer | ⏸️ not run (run stopped) | - | - | - | - | - | - | Rep99 run halted before reaching this stage. |
| logistic_hazard | ⏸️ not run (run stopped) | - | - | - | - | - | - | Rep99 run halted before reaching this stage. |
| rnnsurv | ⏸️ not run (run stopped) | - | - | - | - | - | - | Rep99 run halted before reaching this stage. |

---
**Last Updated:** 2026-08-20 (rep99 terminated by user request; cox result stands, dynamic_deephit/hazard_transformer/logistic_hazard/rnnsurv not completed)
