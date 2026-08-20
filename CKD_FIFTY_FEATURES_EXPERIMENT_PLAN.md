# CKD Fifty Features Heterogeneous Experiment Plan

**Last Updated:** 2026-08-20 18:10 CDT

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
| rep5 | ❌ Terminated | ⏳ | ⏳ | ⏳ | ⏳ | **Terminated by user request 17:43 CDT (Aug 20)** — `cox` had completed successfully (C-index 0.441, IBS 0.923, AUC 0.4345, `.dill` written), and the run was 2d01h+ into `dynamic_deephit` with zero trial-completion output the whole time, consistent with the known `combine_loss` nested-loop bottleneck (see rep1's row) still not fixed. Killed via `kill -TERM -968997` (process group; wrapper 968997 and `dynamic_deephit` child 1077896 shared this PGID) — confirmed both PIDs gone via `ps` re-check immediately after. Session auto-check cron job `ca5a55c0` cancelled — no further auto-monitoring for this rep. `cox` model/results from before termination are preserved in `generated_data/rep5/`; `dynamic_deephit` never produced a `.pt`. Not relaunched — awaiting user direction (e.g. apply the `combine_loss` fix first, then rerun `dynamic_deephit` only). Owned by this session. **CORRECTION (this session, 17:47 CDT Aug 20, verified directly via `ps`/`/proc` on `sunlab-serv-02.cs.illinois.edu`):** an active, still-running `cox` process for rep5 exists on this host — wrapper PID 1776250 (`run_rep.sh 5`) and child PID 1776284 (`python -m pkgs.experiments.cox`), started Mon Aug 17 18:55:10 CDT 2026, elapsed 2d22h50m+, ~1244% CPU / 21.3% MEM, stdout/stderr both pointed at `pkgs/scripts/eval_all_rep5.log`. This is a different PID pair than 968997/1077896 killed above — those PIDs do not exist on sunlab-serv-02. `generated_data/rep5/ckd_fifty_features_heterogeneous_cox_model.dill` (407MB) was already written Aug 18 16:20 CDT, so the fit stage finished; the process is most likely still in the evaluation stage (time-dependent AUC/Brier over the 8.1M-row test panel) — consistent with the CoxTimeVaryingFitter-at-scale slowness already noted for rep2/rep3. All 6 GPUs on sunlab-serv-02 are idle (0% util), so no GPU-stage model is running concurrently for this rep on this host. **Not killed by this session** — still running as of this check; the termination noted above may refer to a different host/run and was left as-is since it wasn't reproduced here. A 10-min recurring check was scheduled (session cron job `89d70466`) to track it going forward. **UPDATE (this session, 17:53 CDT Aug 20, sunlab-serv-02):** by explicit user request, child PID 1776284 (`cox`) was killed via `kill -TERM` and confirmed gone. However wrapper PID 1776250 (`run_rep.sh 5`) treated this as a stage failure (logged `✗ cox failed with exit code 143`) and **auto-advanced to the next stage**, spawning a new `dynamic_deephit` process — **PID 2383419**, started 17:52:03 CDT, still running as of this check (85% CPU / 3.7% MEM). The wrapper itself (1776250) was *not* killed. User has been asked whether to also kill 2383419 and the wrapper; awaiting response. Recurring check `89d70466` continues, now tracking wrapper 1776250 + whichever child it currently owns rather than the original cox PID. **UPDATE (this session, 18:10 CDT Aug 20, sunlab-serv-02):** PID 2383419 (`dynamic_deephit`) died on its own — `torch.OutOfMemoryError` on `cuda:7` during Optuna trial 0's LSTM forward pass (`features` shape `[256, 13713, 100]`; "GPU 7 has a total capacity of 10.57 GiB ... 10.56 GiB memory in use"). This is a genuinely different failure mode than the multi-day `combine_loss`-hang previously documented for rep1/rep4/other rep5 attempts — this scenario's per-patient sequence length (~13,713 timesteps × 100 features, batch 256) simply doesn't fit an 11GB RTX 2080 Ti for this model. Wrapper 1776250 auto-advanced again, now running **`hazard_transformer`** — **PID 2386742**, started 18:09:40 CDT, on `cuda:7`. All 8 GPUs on sunlab-serv-02 (indices 0–7, not 6 as earlier noted) are otherwise idle. User has not yet responded on whether to stop the cascade; flagged again this check since the failure mode changed. |

### Feature Verification (this session, read-only analysis, not part of Phase 1/2)

User asked to verify that the 50 lab features used in
`ckd_fifty_features_heterogeneous_test_data.csv` are actually the 50 most
common CKD-related lab measurements (as opposed to a hand-picked/asserted
list). `pkgs/data_analysis/esrd_lab_analysis.py` only empirically validates
the **top 10** (see `generated_data/rep1/esrd_lab_analysis_report.txt`,
top-9 non-creatinine items all match the CSV's egfr/urea_nitrogen/hemoglobin/
potassium/sodium/bicarbonate/chloride/anion_gap/hematocrit/platelet_count);
items 11–50 in `pkgs/commons.py`/`time_series_store.py` were never checked
against actual `itemid` frequency in `labevents.csv`. Running an independent
frequency count to check.

- **Host**: sunlab-serv-01.cs.illinois.edu
- **PID**: 724331
- **Command**: `python3 /tmp/claude-1244801/-home-minhn2-uiuc-kidney-failure/792c2ac7-f962-49d1-9ab8-5cf0dc53e956/scratchpad/check_top50_labs.py`
- **Log**: `/tmp/claude-1244801/-home-minhn2-uiuc-kidney-failure/792c2ac7-f962-49d1-9ab8-5cf0dc53e956/scratchpad/check_top50_labs.log`
- **Output**: `/tmp/claude-1244801/-home-minhn2-uiuc-kidney-failure/792c2ac7-f962-49d1-9ab8-5cf0dc53e956/scratchpad/itemid_counts.csv`
- **Start time**: 17:56 CDT Aug 20, 2026
- **What it does**: streams `data/mimic-iv-2.2/hosp/labevents.csv` (13.7GB, read-only, `usecols=['subject_id','itemid']`, 5M-row chunks) filtered to the same cohort `get_time_series_data_ckd_patients` uses for this scenario (CKD stage 3–5 + ESRD ICD codes), and tallies `itemid` frequency to compare against the 50 concepts actually selected.
- **Dependencies/blocking**: none — read-only against `data/mimic-iv-2.2/`, does not touch `current_rep`, `generated_data/`, or any file another session's rows in this doc depend on. Not part of the Phase 1/2 pipeline tracked above.
- **Status**: ✅ Complete (17:59 CDT Aug 20). Ran to completion in ~4 min, no errors. PID 724331 no longer running.

**Result: the 50 selected features are NOT exactly the true empirical top 50.** 45/50 land in the
true top 50 by lab-event frequency for the cohort `get_time_series_data_ckd_patients` uses
(CKD stage 3–5 + ESRD ICD codes, n=34,332, 55.1M lab-event rows). Ranks 1–31 match exactly. 5 of the
50 selected concepts rank *below* 50 and should not really be there: `bnp` (44,066 events, rank 52),
`bilirubin_indirect` (43,122, rank 53), `amylase` (38,244, rank 55), `crp` (38,086, rank 56), and
especially `ggt` (only 6,635 events — rank 65, ~7x below the true #50 cutoff). Conversely 4 more-common
concepts were left out of the 50: `proteins` (303,830 events, would rank 32), `tsh` (87,018, rank 38),
`hba1c` (67,985, rank 43), `fibrinogen` (62,647, rank 45) — each more frequent than several labs that
were kept (e.g. `troponin`, `ck`, `lipase`, `triglycerides`, `total_protein`, `cholesterol_total`).
(A 5th candidate, `prealbumin` in `pkgs/commons.py:122`, is a dead/vestigial variable pointing at the
same itemid `50976` as `total_protein` — "Using total protein as proxy" — never referenced anywhere else
in the codebase, so it isn't a genuinely distinct competing feature and was excluded from this count.)

Root cause: `pkgs/data_analysis/esrd_lab_analysis.py::analyze_top_lab_measurements` only ever computed
and saved the **top 10** (see `generated_data/rep1/esrd_lab_analysis_report.txt`) — those 10 all
correctly appear in the 50. The remaining 40 in `pkgs/commons.py`/`time_series_store.py` were added
by hand/domain judgment (per the file comment "50 most common CKD-related labs") but were never
checked against actual `itemid` frequency counts beyond that top 10, so the drift above went
undetected. Net effect on the CSV's feature count is unaffected (still exactly 50 lab values + 50
`_missing` indicators, verified across all of rep1–rep5 and rep99 at 106 columns each), but the claim
that all 50 are literally "the 50 most common" is not accurate as-is.

Reproduction artifacts (session-scratch, not committed): `check_top50_labs.py` /
`itemid_counts.csv` / `compare_top50.py`, all under
`/tmp/claude-1244801/-home-minhn2-uiuc-kidney-failure/792c2ac7-f962-49d1-9ab8-5cf0dc53e956/scratchpad/`.
Not persisted to the repo — this was a one-off verification, not a pipeline change. No fix has been
applied to `pkgs/commons.py`/`time_series_store.py`; swapping the 5 low-frequency features for the 4
(or 5, including a genuine prealbumin lab if one is added later) higher-frequency ones would require
regenerating train/test data for every rep and would invalidate any results already computed for
reps whose data predates the fix — flagging for the user to decide rather than doing unilaterally.

### Phase 3: Results

| Metric | Rep1 | Rep2 | Rep3 | Rep4 | Rep5 | Mean ± Std |
|--------|------|------|------|------|------|------------|
| C-index | - | - | - | - | - | - |
| Brier Score | - | - | - | - | - | - |
| AUC | - | - | - | - | - | - |
