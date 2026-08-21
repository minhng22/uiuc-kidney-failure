# CKD Fifty Features Heterogeneous Experiment Plan

**Last Updated:** 2026-08-20 (feature list changed to match rep5 esrd_lab_analysis_report.txt, sunlab-serv-01 — see "Feature List Change" section below; ⚠️ data/models for reps 1-5 and rep99 now predate this change, see that section; also: prediction horizon changed 1yr→2yr, sunlab-serv-02 — see "Prediction Horizon Change" section below, same ⚠️ caveat applies)

## Overview
Run experiments on the `CKD_FIFTY_FEATURES_HETEROGENEOUS` scenario using all survival models across 5 replications, using the existing `run_all_reps.sh` orchestration script.

---

## Current State
- **Data files**: `ckd_fifty_features_heterogeneous_train_data.csv` and `ckd_fifty_features_heterogeneous_test_data.csv` **do not exist** yet in `generated_data/rep{N}/`
- **Current rep**: `current_rep = 5` in [commons.py](pkgs/commons.py)
- **All model modules already configured** to run `CKD_FIFTY_FEATURES_HETEROGENEOUS` in their `__main__` blocks ✓
- **100 features**: 50 lab values + 50 missingness indicators

## Features Used (100 total) — updated 2026-08-20, see "Feature List Change" section
50 lab values: egfr, urea_nitrogen, hemoglobin, serum_albumin, potassium, sodium, bicarbonate, phosphate, calcium, glucose, chloride, anion_gap, hematocrit, platelet_count, wbc, rbc, mcv, mch, mchc, rdw, magnesium, uric_acid, bilirubin_total, alt, ast, alkaline_phosphatase, ldh, iron, total_protein, cholesterol_total, triglycerides, inr, ptt, crp, ferritin, transferrin, tibc, lymphocytes, neutrophils, monocytes, basophils, eosinophils, pt, rdw_sd, lab_h, lab_l, lab_i, urine_specific_gravity, urine_ph, ph

Plus 50 corresponding missingness indicators (_missing suffix)

(Previous list, superseded 2026-08-20: same first 37 through `tibc`, then `lactate, base_excess, pco2, po2, ph, bilirubin_direct, bilirubin_indirect, ggt, amylase, lipase, ck, troponin, bnp`.)

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

### Feature List Change: aligned to rep5 report (2026-08-20, sunlab-serv-01.cs.illinois.edu)

User explicitly directed: use `generated_data/rep5/esrd_lab_analysis_report.txt` as the reference
and make the code's feature list match it. Implemented as a code change (previous entry above was
audit-only; this one applies it).

**What changed** — swapped the 12 lowest-priority (least renal-specific) of the 22 non-matching
features for the 12 report-top-50 items that are both absent from the code and carry a real
numeric `valuenum` in `labevents.csv` (verified directly against the full file, not assumed):
- **Removed**: `lactate`, `base_excess`, `pco2`, `po2`, `bilirubin_direct`, `bilirubin_indirect`,
  `ggt`, `amylase`, `lipase`, `ck`, `troponin`, `bnp` (blood-gas/cardiac/pancreatic/hepatobiliary
  markers, the least CKD-specific of the original 50)
- **Added**: `lymphocytes` (51244), `neutrophils` (51256), `monocytes` (51254), `basophils`
  (51146), `eosinophils` (51200) — WBC differential; `pt` (51274); `rdw_sd` (52172);
  `lab_h`/`lab_l`/`lab_i` (50934/51678/50947 — d_labitems literally labels these "H"/"L"/"I",
  meaning unresolved, but they're populated numeric Chemistry-panel values); `urine_specific_gravity`
  (51498); `urine_ph` (51491)
- **Result**: 40/50 lab features now have an itemid in the report's top 50 (up from 28/50).

**Kept as-is (10), because no report item can replace them without inventing new architecture**:
`uric_acid`, `ldh`, `iron`, `total_protein`, `cholesterol_total`, `triglycerides`, `crp`,
`ferritin`, `transferrin`, `tibc`. The 10 report-top-50 items that would be needed to close this
last gap all fail on data, not just preference — verified directly against `labevents.csv`:
- `Estimated GFR (MDRD equation)` (itemid 50920): **0 of 1,373,686 rows in the entire file have a
  numeric `valuenum`** — this item is dead/always-empty in this MIMIC-IV extract, unusable by any
  encoding.
- `Specimen Type`, `Urine Color`, `Urine Appearance`, `Leukocytes`[urine], `Bilirubin`[urine],
  `Blood`[urine]: 0% `valuenum` coverage in every sample checked — free-text/dipstick codes only
  (e.g. `"NEG"`, `"TR"`, `"SM"`, `"Cloudy"`, `"Amber"`).
- `Glucose`[urine], `Protein`[urine]: <28% `valuenum` coverage, mostly text.
- `Length of Urine Collection`: <2% `valuenum` coverage.

Adding these would require inventing a categorical/ordinal string-value encoder that doesn't exist
anywhere else in this codebase (every other `get_*_df` function in `store.py` reads `valuenum`
directly) — a real design decision, not done unilaterally. Flagging in case the user wants that
built later.

**Files changed**: [pkgs/commons.py](pkgs/commons.py) (12 new `lab_codes_*`), 
[pkgs/data_analysis/store.py](pkgs/data_analysis/store.py) (12 new `get_*_df` functions, same
pattern as existing ones; old functions for the 12 removed labs left in place, unused, not
deleted), [pkgs/data_analysis/time_series_store.py](pkgs/data_analysis/time_series_store.py)
(`all_labs`, `lab_functions`, `ckd_fifty_cols` ×2, `get_final_columns`, `get_feature_columns`),
and 4 duplicate hardcoded copies of the same 50-lab list that would otherwise have gone out of
sync: [pkgs/experiments/utils.py](pkgs/experiments/utils.py) (`get_tv_rnn_model_features`, used by
`rnnsurv.py`), [pkgs/experiments/dynamic_deephit.py](pkgs/experiments/dynamic_deephit.py),
[pkgs/experiments/hazard_transformer.py](pkgs/experiments/hazard_transformer.py),
[pkgs/experiments/logistic_hazard.py](pkgs/experiments/logistic_hazard.py). `cox.py` needed no
change — it consumes the dataframe's columns directly rather than a hardcoded name list.

**Verified**: all edited files parse (`ast.parse`); `get_feature_columns`/`get_final_columns`
return exactly 50/105 columns with no duplicate names; the 4 hardcoded `lab_names` copies are
byte-identical (as ordered lists) to the canonical list in `utils.py`; re-ran the itemid-match
script against the report and got 40/50, matching the 12-for-12 swap exactly. **Not run**: the
actual data-generation/training pipeline (`get_time_series_data_ckd_patients`,
`model_data_store.py`) — that reads the full `labevents.csv` and takes real time, and touches
`current_rep`/shared state, so it wasn't run in this pass.

**⚠️ Impact on existing data — not yet acted on, flagging for the user:**
`generated_data/rep1/` through `rep5/`'s `ckd_fifty_features_heterogeneous_*.csv` files and every
model trained on them (see Phase 2 table below), plus the `rep99` mini run (see
`CKD_FIFTY_FEATURES_mini_experiment_plan.md`), were all built against the **old** 50-feature
schema. They now predate this code change and are no longer reproducible from current code (the
old code path for the 12 removed labs still exists in `store.py`/`commons.py`, just unreferenced,
so nothing is unrecoverable, but the CSVs on disk have 12 columns that no longer match what the
code would generate today). Regenerating any rep's data/models is a separate, real-time,
`current_rep`-touching operation this session has not performed — needs the user's go-ahead
before running, per the "don't clobber shared state" rule.

### Prediction Horizon Change: 1 year → 2 years (2026-08-20, sunlab-serv-02.cs.illinois.edu)

User explicitly directed: change all models to predict a 2-year (730-day) horizon instead of
1-year (365-day). Applied as a code change only — no data regeneration, no rep touched.

**What changed** — every hardcoded `365`-day evaluation/horizon constant found across the model
and experiment code was changed to `730`:
- Evaluation `times` arrays used for time-dependent AUC / Brier score (`np.arange(1, 365, 1)` →
  `np.arange(1, 730, 1)`, or the `min(365, ...)` guarded variant → `min(730, ...)`) in
  [pkgs/experiments/cox.py](pkgs/experiments/cox.py) (both `run_tv_cox_model`/`run_ti_cox_model`
  call sites), [pkgs/experiments/deepsurv.py](pkgs/experiments/deepsurv.py),
  [pkgs/experiments/gbsa.py](pkgs/experiments/gbsa.py) (both call sites),
  [pkgs/experiments/srf.py](pkgs/experiments/srf.py),
  [pkgs/experiments/weibul.py](pkgs/experiments/weibul.py),
  [pkgs/experiments/rnnsurv.py](pkgs/experiments/rnnsurv.py),
  [pkgs/experiments/survival_svm.py](pkgs/experiments/survival_svm.py),
  [pkgs/experiments/dynamic_deephit.py](pkgs/experiments/dynamic_deephit.py),
  [pkgs/experiments/hazard_transformer.py](pkgs/experiments/hazard_transformer.py) (the `auc()`
  function's `times`), and the shared `compute_brier_score_from_risk_scores` helper in
  [pkgs/experiments/utils.py](pkgs/experiments/utils.py).
- `HazardTransformer`'s internal `self.max_time` (the discretized follow-up horizon its
  `num_time_bins=100` bins span, and what its "survival probability by end of horizon" c-index
  metric is computed against) in [pkgs/models/hazard_transformer.py](pkgs/models/hazard_transformer.py):
  `365` → `730`. Note: this constant was itself just changed by another session/host earlier the
  same day from `365 * 15` (15yr) down to `365` (1yr, to match the rest); this edit moves it to
  `730` (2yr) on top of that.
- `DeepONet`'s `predict_risk_score` fixed reference time-point (used to turn its survival curve
  into a single scalar risk score) in [pkgs/models/deeponet.py](pkgs/models/deeponet.py):
  `torch.tensor([[365.0]])` → `torch.tensor([[730.0]])`.

**Deliberately left unchanged** (verified not to be 1-year horizon constants):
- `DynamicDeepHit.pred_times = 365 * 15` in
  [pkgs/models/dynamicdeephit.py](pkgs/models/dynamicdeephit.py) — this sizes the model's internal
  daily-hazard output layer to a 15-year capacity, already far past 2 years; not a 1-year cutoff to
  fix.
- `t / 365.0` in `compute_brier_score_from_risk_scores`
  ([pkgs/experiments/utils.py:88](pkgs/experiments/utils.py)) — a days→years unit-rate conversion
  inside the exponential-decay formula `S(t) = exp(-lambda * t/365)`, not a horizon cutoff.
- `LogisticHazard`'s `LabTransDiscreteTime(num_durations=50)` bins and `RNNSurv`'s
  `time_intervals` grid — both already derive their span from the observed training-data duration
  range rather than a fixed 365-day cap, so they already extend past 2 years; only their separate
  post-hoc evaluation `times` arrays (fixed 365-day arrays used for AUC/Brier) needed the change,
  and `RNNSurv`'s was included above (`LogisticHazard`'s AUC uses data-driven percentiles of
  observed event times, not a fixed-day array, so nothing there needed changing).

**Verified**: all 12 edited files parse (`ast.parse`); `grep -rn "365"` across
`pkgs/experiments/*.py`/`pkgs/models/*.py` now only matches the two deliberately-unchanged lines
above. **Not run**: no training/eval pipeline was executed — this is a source change only. This
does not affect the `hazard_transformer` process (PID 2386742) or wrapper (PID 1776250) currently
running on this host for rep5 (already documented in the Phase 2 table above, owned by another
session) — it already loaded the old `max_time=365` module into memory before this edit landed, so
its current run is unaffected; only processes started after this edit will pick up the 2-year
horizon.

**⚠️ Impact on existing/running work — not yet acted on, flagging for the user:** same caveat as
the "Feature List Change" section above — any rep's data/models generated or trained before this
edit (including the `hazard_transformer` run currently in progress on this host, PID 2386742) used
the 1-year horizon and now predate this change. Regenerating/retraining is a separate, real-time,
`current_rep`-touching operation not performed by this session — needs the user's go-ahead, and
should be coordinated with whichever session owns the currently-running rep5 cascade before
restarting anything.

### Phase 3: Results

| Metric | Rep1 | Rep2 | Rep3 | Rep4 | Rep5 | Mean ± Std |
|--------|------|------|------|------|------|------------|
| C-index | - | - | - | - | - | - |
| Brier Score | - | - | - | - | - | - |
| AUC | - | - | - | - | - | - |
