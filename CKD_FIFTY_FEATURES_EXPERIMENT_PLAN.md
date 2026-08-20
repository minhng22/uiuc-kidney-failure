# CKD Fifty Features Heterogeneous Experiment Plan

**Last Updated:** 2026-08-20 17:53 CDT

## Overview
Run experiments on the `CKD_FIFTY_FEATURES_HETEROGENEOUS` scenario using all survival models across 5 replications, using the existing `run_all_reps.sh` orchestration script.

---

## Current State
- **Data files**: `ckd_fifty_features_heterogeneous_train_data.csv` and `ckd_fifty_features_heterogeneous_test_data.csv` **do not exist** yet in `generated_data/rep{N}/`
- **Current rep**: `current_rep = 5` in [commons.py](pkgs/commons.py)
- **All model modules already configured** to run `CKD_FIFTY_FEATURES_HETEROGENEOUS` in their `__main__` blocks ✓
- **100 features**: 50 lab values + 50 missingness indicators

## Features Used (100 total)
50 lab values: egfr, urea_nitrogen, hemoglobin, serum_albumin, potassium, sodium, bicarbonate, phosphate, calcium, glucose, chloride, anion_gap, hematocrit, platelet_count, wbc, rbc, mcv, mch, mchc, rdw, magnesium, uric_acid, bilirubin_total, alt, ast, alkaline_phosphatase, ldh, iron, total_protein, cholesterol_total, triglycerides, inr, ptt, crp, ferritin, transferrin, tibc, lactate, base_excess, pco2, po2, ph, bilirubin_direct, bilirubin_indirect, ggt, amylase, lipase, ck, troponin, bnp

Plus 50 corresponding missingness indicators (_missing suffix)

---

## Execution Plan

### Phase 1: Data Generation (for all 5 reps)
Data must be generated for each replication before running experiments.

| Step | Command | Output |
|------|---------|--------|
| 1.1 | `python pkgs/scripts/update_rep.py 1` | Update commons.py → rep1 |
| 1.2 | `python -m pkgs.data_analysis.model_data_store` | Generate rep1 train/test data |
| 1.3 | `python pkgs/scripts/update_rep.py 2` | Update commons.py → rep2 |
| 1.4 | `python -m pkgs.data_analysis.model_data_store` | Generate rep2 train/test data |
| 1.5 | `python pkgs/scripts/update_rep.py 3` | Update commons.py → rep3 |
| 1.6 | `python -m pkgs.data_analysis.model_data_store` | Generate rep3 train/test data |
| 1.7 | `python pkgs/scripts/update_rep.py 4` | Update commons.py → rep4 |
| 1.8 | `python -m pkgs.data_analysis.model_data_store` | Generate rep4 train/test data |
| 1.9 | `python pkgs/scripts/update_rep.py 5` | Update commons.py → rep5 |
| 1.10 | `python -m pkgs.data_analysis.model_data_store` | Generate rep5 train/test data |

### Phase 2: Run All Experiments via run_all_reps.sh
Use existing orchestration script to run all models across all reps.

```bash
cd /home/minhn2/uiuc-kidney-failure
bash pkgs/scripts/run_all_reps.sh --background
```

**What `run_all_reps.sh` does:**
- Iterates through rep1 → rep5
- For each rep:
  - Calls `update_rep.py` to switch data paths
  - Runs 5 models sequentially: cox → dynamic_deephit → hazard_transformer → logistic_hazard → rnnsurv
- Logs output to `pkgs/scripts/eval_all_rep{N}.log`

### Phase 3: Compile Results Report
After all experiments complete, generate summary report from logs.

---

## Output Artifacts (per replication)
| Artifact | Path Pattern |
|----------|--------------|
| Train Data | `generated_data/rep{N}/ckd_fifty_features_heterogeneous_train_data.csv` |
| Test Data | `generated_data/rep{N}/ckd_fifty_features_heterogeneous_test_data.csv` |
| Cox Model | `generated_data/rep{N}/ckd_fifty_features_heterogeneous_cox_model.dill` |
| Dynamic DeepHit | `generated_data/rep{N}/ckd_fifty_features_heterogeneous_ddh_model.pt` |
| Hazard Transformer | `generated_data/rep{N}/ckd_fifty_features_heterogeneous_hazard_transformer_model.pt` |
| Logistic Hazard | `generated_data/rep{N}/ckd_fifty_features_heterogeneous_logistic_hazard_model.pt` |
| RNN Surv | `generated_data/rep{N}/ckd_fifty_features_heterogeneous_rnn_surv_model.pt` |
| Experiment Log | `pkgs/scripts/eval_all_rep{N}.log` |

---

## Metrics Collected (same as existing experiments)
- **Concordance Index (C-index)**: Measures ranking accuracy
- **Integrated Brier Score**: Calibration metric
- **Time-dependent AUC**: Discrimination at various time points

---

## Estimated Time
- Data generation per rep: ~15-30 minutes
- Total data generation (5 reps): ~1.5-2.5 hours
- Each model per rep: 30-60 minutes
- Total experiments (5 models × 5 reps): ~12-25 hours

---

## Progress Tracking (Updated during execution)

### Phase 1: Data Generation

| Rep | Status | Start Time | End Time | Notes |
|-----|--------|------------|----------|-------|
| rep1 | ✅ Complete | May 23, 2026 | May 23, 2026 | Train: 26,277 patients (8.1M records), Test: 6,570 patients (2.1M records) |
| rep2 | ✅ Complete | May 23, 2026 | May 23, 2026 | Data generated successfully |
| rep3 | ✅ Complete | May 23, 2026 | May 24, 2026 | Data generated successfully |
| rep4 | 🔄 Restarted | May 24, 2026 | - | PID 1129693, restarted 16:16 PDT |
| rep5 | ✅ Complete | - | - | Previously generated |

### Phase 2: Model Training (via run_all_reps.sh)

| Rep | Cox | DDH | HazardTrans | LogHazard | RNNSurv | Status |
|-----|-----|-----|-------------|-----------|---------|--------|
| rep1 | ✅ (C-index 0.441) | ❌ Terminated | — | — | — | **Terminated by explicit user request at 17:41 CDT (Aug 20)**, on `sunlab-serv-01.cs.illinois.edu`: `kill -TERM` sent to wrapper PID 4080938 (`run_all_reps.sh`) and stage PID 4084450 (`dynamic_deephit`); both confirmed dead within 3s, no orphaned children. `cox` result (C-index 0.441) stands from before termination. `dynamic_deephit` never completed a single Optuna trial in ~3d21h/45+ CPU-days — root cause confirmed as the `combine_loss` O(batch²×pred_times) bug in `pkgs/experiments/utils.py` (see prior notes/rep4's row); a vectorized, numerically-verified fix exists but has not been applied to the repo. hazard_transformer/logistic_hazard/rnnsurv never ran for this rep. `pkgs/commons.py` `current_rep` was found hardcoded to literal `1` (not the env-var pattern) as of the last check before termination — worth someone confirming/reverting that separately, not done by this session. Cron job `834cccdc` (auto-check) stopped — nothing left running for this session to monitor. |
| rep2 | ❌ Terminated | ⏳ | ⏳ | ⏳ | ⏳ | **Terminated by user request 17:40 CDT (Aug 20)** after user was told the ~2d23h `cox` runtime was not typical for `CoxTimeVaryingFitter` at this scale (8.1M-row time-varying panel, 100 covariates) — this lifelines class scales poorly past ~hundreds-of-thousands of rows, so multi-day/never-finishing runtime was the likely cause rather than a hang. Killed via `kill -TERM -1773295` (process group; setsid-detached, so this cleanly took down the `cox` child too); confirmed both PIDs (1773295 wrapper, 1773328 `cox`) gone via `ps` re-check. Session auto-check cron job `5903a026` cancelled — no further auto-monitoring for this rep. No model saved (`cox` never completed), `generated_data/rep2/` unaffected otherwise. Not relaunched — awaiting user direction on next steps (e.g. a faster Cox implementation, subsampling, or skipping Cox for this scenario). Owned by this session. |
| rep3 | ❌ Terminated | ⏳ | ⏳ | ⏳ | ⏳ | **Terminated by user request 17:40 CDT (Aug 20)**, same reasoning and same action as rep2's row. Killed via `kill -TERM -1773523`; confirmed both PIDs (1773523 wrapper, 1773554 `cox`) gone via `ps` re-check. Cron job `5903a026` (shared with rep2) cancelled. Not relaunched — awaiting user direction. Owned by this session. |
| rep4 | ✅ (C-index 0.441) | ❌ Terminated | — | — | — | **Terminated by explicit user request at 17:41 CDT (Aug 20)**, on `sunlab-serv-01.cs.illinois.edu`: `kill -TERM` sent to wrapper PID 93388 (`run_rep4.sh`) and stage PID 507697 (`dynamic_deephit`); both confirmed dead within 3s, no orphaned children. `cox` result (C-index 0.441, IBS 0.923, AUC 0.4345, completed 20:42:40 CDT Aug 19) stands from before termination. `dynamic_deephit` ran ~20h43m / 403 CPU-hours with zero completed Optuna trials before termination — same `combine_loss` root cause as rep1 (see that row). hazard_transformer/logistic_hazard/rnnsurv never ran for this rep. Isolated copy at `/home/minhn2/kidney-rep4-run` left in place (untouched, not deleted). Cron job `46731da2` (auto-check) stopped — nothing left running for this session to monitor. |
| rep5 | ❌ Terminated | ⏳ | ⏳ | ⏳ | ⏳ | **Terminated by user request 17:43 CDT (Aug 20)** — `cox` had completed successfully (C-index 0.441, IBS 0.923, AUC 0.4345, `.dill` written), and the run was 2d01h+ into `dynamic_deephit` with zero trial-completion output the whole time, consistent with the known `combine_loss` nested-loop bottleneck (see rep1's row) still not fixed. Killed via `kill -TERM -968997` (process group; wrapper 968997 and `dynamic_deephit` child 1077896 shared this PGID) — confirmed both PIDs gone via `ps` re-check immediately after. Session auto-check cron job `ca5a55c0` cancelled — no further auto-monitoring for this rep. `cox` model/results from before termination are preserved in `generated_data/rep5/`; `dynamic_deephit` never produced a `.pt`. Not relaunched — awaiting user direction (e.g. apply the `combine_loss` fix first, then rerun `dynamic_deephit` only). Owned by this session. **CORRECTION (this session, 17:47 CDT Aug 20, verified directly via `ps`/`/proc` on `sunlab-serv-02.cs.illinois.edu`):** an active, still-running `cox` process for rep5 exists on this host — wrapper PID 1776250 (`run_rep.sh 5`) and child PID 1776284 (`python -m pkgs.experiments.cox`), started Mon Aug 17 18:55:10 CDT 2026, elapsed 2d22h50m+, ~1244% CPU / 21.3% MEM, stdout/stderr both pointed at `pkgs/scripts/eval_all_rep5.log`. This is a different PID pair than 968997/1077896 killed above — those PIDs do not exist on sunlab-serv-02. `generated_data/rep5/ckd_fifty_features_heterogeneous_cox_model.dill` (407MB) was already written Aug 18 16:20 CDT, so the fit stage finished; the process is most likely still in the evaluation stage (time-dependent AUC/Brier over the 8.1M-row test panel) — consistent with the CoxTimeVaryingFitter-at-scale slowness already noted for rep2/rep3. All 6 GPUs on sunlab-serv-02 are idle (0% util), so no GPU-stage model is running concurrently for this rep on this host. **Not killed by this session** — still running as of this check; the termination noted above may refer to a different host/run and was left as-is since it wasn't reproduced here. A 10-min recurring check was scheduled (session cron job `89d70466`) to track it going forward. **UPDATE (this session, 17:53 CDT Aug 20, sunlab-serv-02):** by explicit user request, child PID 1776284 (`cox`) was killed via `kill -TERM` and confirmed gone. However wrapper PID 1776250 (`run_rep.sh 5`) treated this as a stage failure (logged `✗ cox failed with exit code 143`) and **auto-advanced to the next stage**, spawning a new `dynamic_deephit` process — **PID 2383419**, started 17:52:03 CDT, still running as of this check (85% CPU / 3.7% MEM). The wrapper itself (1776250) was *not* killed. User has been asked whether to also kill 2383419 and the wrapper; awaiting response. Recurring check `89d70466` continues, now tracking wrapper 1776250 + whichever child it currently owns rather than the original cox PID. |

### Phase 3: Results

| Metric | Rep1 | Rep2 | Rep3 | Rep4 | Rep5 | Mean ± Std |
|--------|------|------|------|------|------|------------|
| C-index | - | - | - | - | - | - |
| Brier Score | - | - | - | - | - | - |
| AUC | - | - | - | - | - | - |
