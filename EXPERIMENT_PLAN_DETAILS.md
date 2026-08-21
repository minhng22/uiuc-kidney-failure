# Experiment Plan Details (Stage 0 output — awaiting approval)

Detailed execution plan for the three experiments defined in
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): **4 features**, **8 features**, **20 features**.
This doc follows the existing per-scenario pattern already used for
`egfr_components`, `fivelabms`, `heterogeneous`, and `ckd_fifty_features_heterogeneous`
(see [pkgs/commons.py](pkgs/commons.py), [pkgs/data_analysis/types.py](pkgs/data_analysis/types.py),
[pkgs/data_analysis/time_series_store.py](pkgs/data_analysis/time_series_store.py),
[pkgs/data_analysis/model_data_store.py](pkgs/data_analysis/model_data_store.py)).

**Do not edit [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) — it is marked "DON'T EDIT".**

## New scenario naming

| Experiment | `ExperimentScenario` enum value | Feature list (row format) |
|---|---|---|
| 4 features | `FOUR_FEATURES` = `"four_features"` | `age, gender, egfr, uacr` (no missingness flags — like `EGFR_COMPONENTS`) |
| 8 features | `EIGHT_FEATURES` = `"eight_features"` | `age, gender, egfr, uacr, calcium, phosphate, bicarbonate, serum_albumin` (no missingness flags) |
| 20 features | `TWENTY_FEATURES_HETEROGENEOUS` = `"twenty_features_heterogeneous"` | 20 labs, each as `<value>, <value>_missing` — same pattern as `CKD_FIFTY_FEATURES_HETEROGENEOUS` |

`uacr` (Urine Albumin/Creatinine Ratio) is not currently extracted anywhere in the
codebase. Confirmed MIMIC-IV has a direct ratio lab item — itemid `51070`
("Albumin/Creatinine, Urine"), 54,939 records in `labevents.csv` — so no manual
ratio computation from separate albumin/creatinine values is needed.
`lab_codes_albumin = ['51069', '51070', '52703']` in `commons.py` must be left
untouched (used as-is by existing scenarios). A new, separate
`lab_codes_uacr = ['51070']` constant will be added alongside it for the new
`four_features`/`eight_features` scenarios' UACR lookup — `51070` will simply
appear in both lists.

## KFRE baseline model (4 features / 8 features only)

Both of these experiments exist specifically to benchmark against the published
Kidney Failure Risk Equation (Tangri et al.), so **KFRE will be added as an
additional model** alongside `cox`/`dynamic_deephit`/`hazard_transformer`/
`logistic_hazard`/`rnnsurv` — but only for `four_features` and `eight_features`
(not `twenty_features_heterogeneous`, which has no published KFRE formula).

- KFRE is a **closed-form clinical equation, not a trained model**: risk score
  is computed directly from the published 4-variable and 8-variable coefficients
  (Tangri et al. 2011/2016), using each row's `age, gender, egfr, uacr`
  (+ `calcium, phosphate, bicarbonate, serum_albumin` for the 8-variable version).
  No `.fit()`/training step, so no `*_model.pt`/`.dill` artifact needs saving —
  only the computed per-row risk scores (cached as
  `<scenario>_kfre_risk_scores.csv` under `generated_data/rep<N>/` so repeat
  evaluation runs don't recompute them).
- New file: [pkgs/experiments/kfre.py](pkgs/experiments/kfre.py) with a
  `run_kfre_model(scenario)` entry point (mirroring `run_cox_model`'s shape:
  load train/test data via `get_train_test_data(scenario)`, compute risk scores
  for the test set, then report the same metrics as the other models —
  concordance index, time-dependent AUC, Brier score — for apples-to-apples
  comparison).
- **4-variable coefficients (confirmed, cross-checked against two independent sources):**
  `L = -0.2201×(age/10 − 7.036) + 0.2467×(male − 0.5642) − 0.5567×(eGFR/5 − 7.222) + 0.4510×(ln(uACR mg/g) − 5.137)`,
  `Risk(t) = 1 − S0(t)^exp(L)`. Using **North American** calibration (this is a US-based
  MIMIC-IV cohort): S0(2yr) = 0.9750. Only `S0(2yr)` is needed since the horizon is 2 years.
- **8-variable coefficients: DECIDED — use whatever the original published paper (Tangri et al.)
  used.** See Stage 1a Task B below — the exact table must be located and cited (paper + section/table)
  before this branch of `kfre.py` is written. No invented/approximated coefficients will be used.
- `pkgs/scripts/run_rep.sh`'s `EXPERIMENTS` array (and the rep99 mini-experiment
  runner) gains `kfre` alongside the other 5, but `kfre` is skipped for the
  `twenty_features_heterogeneous` scenario.

## Stage 1: Build train/test datasets

### 1a. Pre-implementation research (run in parallel, before any code changes — 1b/1c depend on these outputs)

**Task A — Determine the 20 lab features.**
- Run `analyze_top_lab_measurements(patients_df, labevents_df, d_labitems_df, top_n=20)` from
  [pkgs/data_analysis/esrd_lab_analysis.py](pkgs/data_analysis/esrd_lab_analysis.py) as-is (its existing
  `load_data()` cohort, `get_ckd_patients_and_diagnoses(late_stage=True)`) to get the ranked top-20 itemid
  list, and **use those 20 features directly** — no re-litigating cohort choice or hand-curating swaps
  the way `ckd_fifty_features_heterogeneous` did.
- Cross-reference each of the 20 itemids against the `get_*_df` helpers that already exist in
  `time_series_store.py` (most will already exist, since the 50-feature scenario's top labs by frequency
  mostly overlap). Write new `get_*_df` helpers only for itemids not already covered.
- Record the final 20-feature list + itemid mapping + coverage stats in a short report, saved as
  `generated_data/rep1/twenty_features_lab_analysis_report.txt` (same format/location convention as
  `esrd_lab_analysis_report.txt`).

**Task B — Locate the original Tangri et al. paper's 8-variable KFRE coefficients.**
- Find and cite the source paper(s) that back up the 8-variable coefficients used — the original
  derivation (Tangri N, et al. "A Predictive Model for Progression of Chronic Kidney Disease to Kidney
  Failure." JAMA. 2011;305(15):1553-1559, PMID 21482743) and/or the multinational recalibration (Tangri N,
  et al. JAMA. 2016;315(2):164-174, PMID 26757465) that also published/validated it.
- Extract the exact 8-variable coefficients (age, sex, eGFR, log(ACR), albumin, calcium, phosphate,
  bicarbonate) and their 2-year baseline survival constant(s), the same way the 4-variable coefficients
  above were cross-checked against independent sources.
- Every coefficient used in `kfre.py` must be traceable to a specific cited paper/section/table — no
  values will be used without a citation backing them.
- Runs in parallel with Task A (independent lookups, no shared dependency).

**Both Task A and Task B stop here and wait for approval before 1b/1c proceed** — 1b's code changes depend
on Task A's confirmed 20-feature list, and `kfre.py`'s 8-variable branch depends on Task B's confirmed
coefficients.

### 1b. Code changes (per scenario, following the `egfr_components` / `ckd_fifty_features_heterogeneous` pattern)
1. [pkgs/commons.py](pkgs/commons.py): add `lab_codes_uacr` (new constant, `lab_codes_albumin` left untouched); add path constants
   (`four_features_train_data_path`, `four_features_test_data_path`, `four_features_*_model_path` for
   cox/ddh/hazard_transformer/logistic_hazard/rnn_surv, and the equivalent for `eight_features` and
   `twenty_features_heterogeneous`).
2. [pkgs/data_analysis/types.py](pkgs/data_analysis/types.py): add the 3 new `ExperimentScenario` enum values above.
3. [pkgs/data_analysis/time_series_store.py](pkgs/data_analysis/time_series_store.py):
   - Add `get_uacr_df` (mirrors existing `get_egfr_df`/`get_calcium_df` style helpers).
   - Add scenario branches in the raw-lab-fetch function (~line 200) for `FOUR_FEATURES`/`EIGHT_FEATURES`
     (plain values, no missingness — like `EGFR_COMPONENTS`) and `TWENTY_FEATURES_HETEROGENEOUS`
     (value + `_missing` pairs — like `CKD_FIFTY_FEATURES_HETEROGENEOUS`, using Task A's 20-feature list).
   - Add matching branches in `get_feature_columns` and in `get_time_series_data_ckd_patients`'s final
     column-selection block.
4. [pkgs/data_analysis/model_data_store.py](pkgs/data_analysis/model_data_store.py): add `get_train_test_data`
   branches for the 3 new scenarios (train/test split + CSV write, same as `EGFR_COMPONENTS`/
   `CKD_FIFTY_FEATURES_HETEROGENEOUS`).
5. [pkgs/experiments/*.py](pkgs/experiments/cox.py) (`cox.py`, `dynamic_deephit.py`, `hazard_transformer.py`,
   `logistic_hazard.py`, `rnnsurv.py`): extend each `assert scenario in [...]` allow-list and `get_model_path`
   switch to include the 3 new scenarios. **Also update each file's `if __name__ == '__main__':` block**
   (currently hardcoded to call only `CKD_FIFTY_FEATURES_HETEROGENEOUS`, per the existing pattern used for
   every prior scenario) to also call `FOUR_FEATURES`, `EIGHT_FEATURES`, `TWENTY_FEATURES_HETEROGENEOUS` —
   this is what `run_rep.sh`'s `python -m pkgs.experiments.<name>` actually executes, so without this edit
   the new scenarios would never run even though `run_rep.sh` itself needs no changes beyond adding `kfre`.
6. New [pkgs/experiments/kfre.py](pkgs/experiments/kfre.py): closed-form 4-/8-variable KFRE risk
   calculator + evaluation (`assert scenario in [FOUR_FEATURES, EIGHT_FEATURES]`), using Task B's
   cited coefficients — see "KFRE baseline model" section above.

### 1c. Data extraction commands (background, per rep, reusing existing scripts): parallel

**No new shell scripts.** Reuse what's already there:
- [pkgs/scripts/run_ckd_fifty_features_extraction.sh](pkgs/scripts/run_ckd_fifty_features_extraction.sh) already
  does exactly this job generically — it just runs `python -m pkgs.data_analysis.model_data_store` in the
  background with logging/PID tracking. It doesn't hardcode a scenario itself; the scenario comes from
  that module's `__main__` (currently hardcoded to `CKD_FIFTY_FEATURES_HETEROGENEOUS`, see 1b item 4 —
  will be updated to loop over `FOUR_FEATURES`, `EIGHT_FEATURES`, `TWENTY_FEATURES_HETEROGENEOUS`).
- The only gap: its PID/log filenames aren't rep-specific (`ckd_fifty_features_extraction.pid`, one fixed
  name), which would collide if 5 reps run this concurrently. **Small in-place edit** to that script:
  make the log/PID filenames include `${CKD_REP:-5}` so concurrent invocations don't clobber each other's
  PID file. No new script file.
- Launch by setting `CKD_REP` before invoking the (unmodified-in-spirit) script, once per rep, all 5 at once:
  ```bash
  for rep in 1 2 3 4 5; do CKD_REP=$rep bash pkgs/scripts/run_ckd_fifty_features_extraction.sh; done
  ```
  (env var is inherited by the script's `nohup bash -c "..."` subshell, so each of the 5 backgrounded runs
  resolves paths via its own `CKD_REP`).
- Every launch recorded (PID, log path, start time, rep) in a new `FEATURE_SET_EXPERIMENT_PLAN.md` at repo
  root, per the repo's background-process tracking rule in
  [copilot-instructions.md](.github/copilot-instructions.md), including hostname (`hostname` command).
- Status/completion is verified by tailing each log and checking the output CSVs exist under
  `generated_data/rep<N>/`.

## Stage 2: Mini-experiment (rep99)

**No new scripts.** Reuse existing ones in place:
- [pkgs/scripts/build_mini_experiment_data.py](pkgs/scripts/build_mini_experiment_data.py): currently
  hardcodes `ckd_fifty_features_heterogeneous`'s train/test paths. Generalize it in place (loop over the
  3 new scenarios' path constants instead of one hardcoded scenario) rather than adding new copies —
  same 250 ESRD + 250 non-ESRD stratified-sample logic, unchanged.
- [pkgs/scripts/run_rep.sh](pkgs/scripts/run_rep.sh) already takes the rep number as a plain argument
  (`run_rep.sh 99` works today, no "rep99-scoped variant" needed) and already sets `CKD_REP`/`PYTHONPATH`
  generically. Only its `EXPERIMENTS` array changes (adding `kfre`, see KFRE section above) — same file,
  no sibling script.
- Run `pkgs/scripts/run_rep.sh 99` to sanity-check the full pipeline (3 new scenarios × 5 ML models, plus
  `kfre` for `four_features`/`eight_features`) end-to-end before committing to full 5-rep runs.

## Stage 2.1: Cohort/feature-importance analysis
- Extend [pkgs/data_analysis/feature_importance_analysis.py](pkgs/data_analysis/feature_importance_analysis.py)
  with `analyze_four_features`/`analyze_eight_features`/`analyze_twenty_features` methods mirroring
  `analyze_egfr_components`/`analyze_fivelabms` (SHAP-based importance per model, using the rep99 or rep1
  trained models).
- Output report saved under `generated_data/rep<N>/<scenario>_shap_analysis_report.txt`, matching existing
  naming (`egfr_components_shap_analysis_report.txt`).

## Stage 3: Full experiment runs (rep1 → rep5)
- Once Stage 2's mini-experiment passes, launch full runs via the **same**
  [pkgs/scripts/run_rep.sh](pkgs/scripts/run_rep.sh) (no new script) for `N` in 1..5 — each in the
  background, each PID/log recorded in `FEATURE_SET_EXPERIMENT_PLAN.md`.
- Per the repo's 10-minute auto-check rule, status will be re-verified periodically (`ps -p <pid>`,
  log tail) and the plan doc updated until all 5 reps finish or fail-and-are-relaunched.

## Open questions before implementation starts
