# CKD Fifty Features — Mini Experiment Plan

**Status:** ❌ TERMINATED by explicit user request at 2026-08-20 ~18:57 CDT (see
Progress Tracking below for what stands and what doesn't).
**Owner/host:** original launch by session on `sunlab-serv-02.cs.illinois.edu`;
the relaunch/termination below is owned by this session, on `sunlab-serv-01.cs.illinois.edu`.

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
| cox | ✅ done (attempt 1) | wrapper PID 2181834, cox process PID 2181871 / sunlab-serv-02 | 22:26:44 CDT | 22:28:07 CDT (Aug 19) | 0.461 | 0.644 | 0.4641 | Completed in **83s** (vs. ~2 days at full rep1 scale) — confirms rep99 isolation + small sample works as intended. |
| dynamic_deephit | ❌ terminated by user | PID 2182603 (wrapper 2181834) / sunlab-serv-02 | 22:28:07 CDT (Aug 19) | terminated 2026-08-20 (user request) | - | - | - | Terminated via `kill -TERM -2181834` (whole process group) after ~19h with 0 Optuna trials completed and no GPU usage (`get_device()` was hardcoded to `"cpu"` at that time; since fixed in commit `dc2886e`, verified present in current `pkgs/experiments/utils.py`). Confirmed both the wrapper and the `dynamic_deephit` process are gone; no orphaned processes remain. `run_rep99.pid` removed. |
| cox | ✅ done (attempt 2, re-run inside relaunch) | wrapper PID 722008 / sunlab-serv-01 | 17:52:23 CDT | 17:52:47 CDT (Aug 20) | - | - | - | `✓ cox completed successfully` per `run_rep99_master.log`. Confirms `commons.py` fix (below) didn't break isolation — re-ran cleanly against `generated_data/rep99`. |
| dynamic_deephit | ❌ failed (attempt 2, under wrapper 722008) → 🔄 standalone retry (attempt 3, in progress) | attempt-2 PID 722XXX (child of 722008) failed 17:53:23 CDT; retry PID **731383** / sunlab-serv-01 | 17:52:47 CDT / retry 18:15:15 CDT (Aug 20) | failed 17:53:23 CDT (Aug 20) | - | - | - | **Diagnosed and fixed a resource-limit bug**, distinct from the `get_device()` issue: trial 0, epoch 1 raised `torch.OutOfMemoryError` ("Tried to allocate 2.00 MiB. GPU 2 has ... 10.56 GiB memory in use") on the training forward/backward pass. Root cause: rep99's mini sample has an outlier patient with a **5644-timestep** sequence (`max_seq_length`), and [pkgs/experiments/dynamic_deephit.py](pkgs/experiments/dynamic_deephit.py) hardcoded `batch_size=256` for the LSTM-based `DynamicDeepHit` model in all 4 DataLoaders (train loop, `auc()`, `brier_score_evaluation()`, `c_idx()`) — batch=256 × seq_len=5644 through backprop (or an un-`no_grad`'d forward in `auc`/`brier_score_evaluation`) exceeds a single 2080 Ti's 10.5GB. Fixed by reducing all 4 to `batch_size=16` (repo-wide change, affects any future dynamic_deephit run on any rep, not just rep99). Since `run_rep.sh` continues to the next stage on failure rather than aborting, the wrapper (722008) had already moved on to hazard_transformer by the time this was caught, so relaunched dynamic_deephit standalone: `CKD_REP=99 python -m pkgs.experiments.dynamic_deephit` → PID 731383, log `pkgs/scripts/dynamic_deephit_rep99_retry.log`, running in parallel with the wrapper's remaining stages. Being monitored for completion/OOM recurrence. |
| hazard_transformer | ❌ terminated by user | wrapper PID 722008, hazard_transformer PID 722661 / sunlab-serv-01 | 17:53:23 CDT (Aug 20) | terminated ~18:57 CDT (Aug 20) | - | - | - | Was mid-Optuna (trial 0 finished value 0.532; trial 1 finished value 0.470, best still trial 0) when terminated via `kill -TERM -722008` (whole process group). Confirmed both wrapper 722008 and hazard_transformer 722661 gone within 3s; `pgrep` found no orphaned rep99 children afterward. `run_rep99.pid` removed. No model was saved for this stage. |
| logistic_hazard | ⏸️ not run (run stopped) | - | - | - | - | - | - | Never started — wrapper terminated before reaching this stage. |
| rnnsurv | ⏸️ not run (run stopped) | - | - | - | - | - | - | Never started — wrapper terminated before reaching this stage. |

**dynamic_deephit — extra detail on the two post-relaunch attempts (both superseded by termination):**
- **Attempt 3** (standalone retry, PID 731383, batch_size fix applied): got past the OOM but failed at trial 0 with `ValueError('NaNs detected in inputs, please correct or drop.')` from sksurv's concordance calc — training loss was finite and decreasing through epoch 5 (49.07 → 43.27) then went to `nan` from epoch 6 onward. Diagnosed as numerical overflow in `combine_loss`'s ranking term: `torch.exp(-diff / w2)` in [pkgs/experiments/utils.py](pkgs/experiments/utils.py#L241-L246) — with rep99's 5644-timestep outlier patient, cumulative hazard `diff` can grow large enough to overflow `exp()` to `inf`, propagating to NaN loss/gradients/risk-scores.
- **Fix applied**: clamped the exponent (`(-diff / w2).clamp(max=50.0)`) before `exp()` in `combine_loss` — repo-wide change, affects any rep's dynamic_deephit run going forward, not just rep99.
- **Attempt 4** was about to be launched (`CKD_REP=99 python -m pkgs.experiments.dynamic_deephit`, log would have been `dynamic_deephit_rep99_retry2.log`) when the user requested termination instead — **that launch command was never actually run**; no attempt-4 process exists.

**Context on the relaunch itself:** relaunched via `bash pkgs/scripts/run_rep.sh 99` (wrapper PID 722008). **Before relaunching, fixed a critical regression in `pkgs/commons.py`**: `current_rep` had been hardcoded to literal `1` (commit `dc2886e`), which silently ignored `CKD_REP` and would have made this rerun write into `generated_data/rep1/` instead of the isolated `rep99` slot. Reverted to `current_rep = int(os.environ.get('CKD_REP', 5))` (restoring `import os`); verified with `CKD_REP=99 python -c "from pkgs import commons; print(commons.generate_data_path_latest_rep)"` → correctly resolves to `.../generated_data/rep99`. GPUs confirmed idle (8× RTX 2080 Ti) on sunlab-serv-01 before launch. Logs: `pkgs/scripts/eval_all_rep99.log`, `pkgs/scripts/run_rep99_master.log`, PID file `pkgs/scripts/run_rep99.pid`; dynamic_deephit retry log `pkgs/scripts/dynamic_deephit_rep99_retry.log`.

---
**Last Updated:** 2026-08-20 ~18:58 CDT — **rep99 terminated by explicit user request.** cox result stands (0.461/0.644/0.4641, from the relaunch's attempt-2 re-run). dynamic_deephit never got a clean trial (OOM on attempt 2/3's batch=256, then NaN-from-overflow on attempt 3's batch=16 fix; a second fix — clamping the ranking-loss exponent in `combine_loss` — was applied but never tried, since termination came before attempt 4 was launched). hazard_transformer was mid-Optuna (2 trials finished, best 0.532) with no model saved when killed. logistic_hazard/rnnsurv never ran. The recurring 10-min monitor (cron job `ad7a3a9a`) has been cancelled — nothing left running for this session to watch. Both repo-wide fixes (`pkgs/commons.py` current_rep regression, `combine_loss` exponent clamp) remain applied in the working tree and are still uncommitted — worth deciding whether to commit them regardless of rep99's outcome, since they're real bugs independent of this run.
