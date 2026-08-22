# Experiment Plan Details (Stage 0 output — awaiting approval)

Detailed execution plan for the three experiments defined in
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): **4 features**, **8 features**, **20 features**.
This doc follows the existing per-scenario pattern already used for
`egfr_components`, `fivelabms`, `heterogeneous`, and `ckd_fifty_features_heterogeneous`
(see [pkgs/commons.py](pkgs/commons.py), [pkgs/data_analysis/types.py](pkgs/data_analysis/types.py),
[pkgs/data_analysis/time_series_store.py](pkgs/data_analysis/time_series_store.py),
[pkgs/data_analysis/model_data_store.py](pkgs/data_analysis/model_data_store.py)).

**Do not edit [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) — it is marked "DON'T EDIT".**
**Create report for each experiment. Update report every time you do something. 
Update high-level progress of stages in `EXPERIMENT_STATUS.md` -  don't swamp this file with details.**

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

### 1a-2. Multi-lab-source row merge design for `four_features`/`eight_features` (resolved 2026-08-21)

## Problem

No existing scenario merges multiple distinct lab-event sources into one row with real,
non-missing, simultaneous values — every other multi-lab scenario (`heterogeneous`, `fivelabms`,
`ckd_fifty_features_heterogeneous`) avoids this via missingness flags instead (one row = one lab type's
event, everything else zeroed/flagged missing), which this scenario must not do.

Two sub-problems fall out of this:
1. **Row format.** KFRE (`risk_pred_core`) is a closed-form, point-in-time equation — one covariate
   snapshot in, one 2yr/5yr risk out, no start/stop notion. Does that mean the extracted data needs a
   non-time-variant format?
2. **Repeat measurements.** A patient can have a lab (e.g. creatinine) drawn multiple times within one
   admission. A naive "anchor once per admission" merge silently collapses those repeats into a single
   row, losing signal and leaving "what happens on repeat measurement" answered by fiat rather than by a
   stated rule.

## Fix (with backing references)

**Row format stays time-variant.** KFRE's formula is agnostic to row packaging and can be applied
row-by-row, treating each row as its own landmark snapshot — consistent with the plan's existing
description of `run_kfre_model` reusing `get_train_test_data(scenario)`'s output as-is. So: **keep the
time-variant start/stop format**, unchanged from what `cox.py`/`dynamic_deephit.py`/
`hazard_transformer.py`/`logistic_hazard.py`/`rnnsurv.py` already expect; `kfre.py` applies its
closed-form equation to each row independently.

**Merge strategy: anchor on each creatinine draw, bounded nearest-value window per lab type** — this is
a well-studied problem (sparse, asynchronously-timed covariates in EHR-derived survival models), and the
rule below follows established practice rather than an invented one:
- **Anchor on each creatinine measurement** (restores per-event granularity, matching `EGFR_COMPONENTS`,
  rather than one row per admission) — compute eGFR from it; this is the row's anchor time.
- **Calcium, phosphate, bicarbonate, serum_albumin** (`eight_features` only, routinely drawn on the same
  chemistry panel as creatinine): nearest value to the anchor, **within ±24 hours**. Backed by APACHE II
  (Knaus WA, Draper EA, Wagner DP, Zimmerman JE. "APACHE II: A Severity of Disease Classification
  System." *Critical Care Medicine*. 1985;13(10):818-829), which treats all labs drawn in a patient's
  first 24 hours as one physiologic snapshot — including serum creatinine, sodium, and potassium, the
  same chemistry panel as here. 48h was considered and rejected: every 48h convention found means
  something different from "concurrent snapshot" — KDIGO AKI staging uses 48h as a *change-detection*
  window (creatinine rising ≥0.3 mg/dL *within* 48h, the opposite concept), and the MIMIC-III benchmark
  (Harutyunyan et al., 2019) uses 48h as a *total lookback observation horizon*, not a matching
  tolerance. No 48h convention analogous to APACHE II's use was found. (The window length itself has no
  single universal number — a systematic review of 92 EHR temporal-modeling studies, Yang et al.,
  "Assessment of Prediction Tasks and Time Window Selection in Temporal Modeling of Electronic Health
  Record Data," PMC10449760, 2023, found window length is consistently chosen per-application; the
  literature's requirement is *that* a bounded window is used and stated, not a specific figure — 24h is
  the one number here that does have direct backing, via APACHE II above.)
- **uACR** (a separately-ordered urine test): nearest value to the anchor. **Originally bounded to
  the same hospital admission** — precedent: Tangri et al. 2016 (`JAMA` 315(2):164-174 — the same
  paper used for the 8-variable coefficients) documents this exact problem for KFRE itself, since
  several of its 31 validation cohorts are real-world EHR data; eAppendix 1 states the resolution
  used for one such cohort verbatim: *"Geisinger: ... Covariates obtained most closely to index date
  within a past year were included in models."* — nearest value to a reference time, bounded by an
  explicit lookback window, not unbounded reuse. General framework: landmarking methodology (van
  Houwelingen, "Dynamic Prediction by Landmarking in Event History Analysis," 2007; Putter & van
  Houwelingen, "Landmarking 2.0," *Statistics in Medicine*, 2022) is the standard way to turn sparse,
  time-varying covariates into fixed values at chosen landmark times; recent EHR work applying it (Hu
  et al., arXiv:2204.05870, 2022) shows plain *unbounded* last-observation-carried-forward is biased
  in sparse EHR settings, supporting a bounded window over reuse across an entire multi-week admission.
  **Loosened after the 1c-0 pilot** (2026-08-21): the same-admission bound caused 96.5% patient
  attrition (34,332 → 1,220 for `four_features`), disproportionately dropping ESRD-negative patients
  (0.8% retained vs. 3.8% for ESRD-positive), skewing the extracted cohort's outcome rate from the
  source population's 91.9% positive to 98.2% positive — see the 1c-0 report for the full numbers.
  Per user direction, the uACR match is now bounded to the **patient's whole history**
  (`by='subject_id'` instead of `by='hadm_id'` in `merge_nearest_within_admission`) rather than one
  admission, trading the tighter same-encounter simultaneity guarantee for a workable cohort size.
  This reintroduces some of the unbounded-LOCF bias risk noted above (a uACR reading could now be
  reused across admissions months or years apart) — a known, accepted tradeoff, not an oversight.
- **Drop, don't flag.** If any required lab has no qualifying value within its window, that
  creatinine-draw row is dropped entirely — no partial rows, no missingness flags.
- **One row per qualifying creatinine draw.** A patient monitored multiple times in one admission
  correctly yields multiple rows, each independently paired with its nearest in-window value of each
  other required lab — so a sparser lab like uACR may legitimately be reused (nearest-value) across
  several same-admission rows. Expected under the landmarking framework above, not a bug.

**Example rows** (`four_features`; column order matches `get_final_columns`:
`subject_id, duration_in_days, start, stop, age, gender, egfr, uacr, has_esrd`). Subject 10023567 has 2
creatinine draws in one admission but only 1 uACR draw that admission, so the same uACR value is
correctly reused across both rows:

| subject_id | duration_in_days | start | stop | age | gender | egfr | uacr | has_esrd |
|---|---|---|---|---|---|---|---|---|
| 10023567 | 412.3 | 412.3 | 415.1 | 67 | 1 | 34.2 | 320.5 | 0 |
| 10023567 | 415.1 | 415.1 | 630.8 | 67 | 1 | 31.8 | 320.5 | 0 |

Two creatinine draws 2.8 days apart in the same admission each anchor their own row; both pair with the
admission's single uACR draw (320.5 mg/g), the nearest — and only — qualifying value for both anchors;
eGFR differs between rows (34.2 vs 31.8) reflecting the actual change in kidney function over 2.8 days;
`has_esrd=0` throughout since the patient wasn't yet diagnosed as of either row.

`eight_features` is the same shape with `calcium, phosphate, bicarbonate, serum_albumin` added before
`has_esrd` (e.g. `..., calcium=8.9, phosphate=4.2, bicarbonate=23, serum_albumin=3.6, has_esrd=0`).

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

### 1c-0. Pilot extraction (rep1 only) + cohort analysis — approval gate before full 1c

Before committing to 5 parallel background extraction runs (1c), run the same extraction **once, for
rep1 only**: `CKD_REP=1 python -m pkgs.data_analysis.model_data_store` (same 1c extraction command,
rep1 only) — then produce the cohort analysis report below and stop for user approval before scaling
out.

**Proposed data analysis** (`generated_data/rep1/<scenario>_cohort_flow_report.txt`, one per new
scenario) — one table, source cohort vs. final extracted cohort as the two columns:

| Analysis | Why | Reference |
|---|---|---|
| n (patients), records | Baseline count needed before any other stat means anything; shows attrition from the merge design | Major et al. 2019, *PLOS Medicine*, Table 1 ("n" row); STROBE (von Elm et al. 2007, *Lancet*) item 13(a) |
| % male | Standard demographic in every KFRE-adjacent cohort table | Major et al. 2019, Table 1 ("Female" row) |
| Mean age, years (SD) | Standard demographic; age is a KFRE predictor | Major et al. 2019, Table 1 |
| Mean / median eGFR (SD / IQR) | Core KFRE predictor; reported both ways in the source literature | Major et al. 2019, Table 1 |
| Mean / median uACR (SD / IQR) | Core KFRE predictor; heavily right-skewed, mean/SD alone would mislead | Major et al. 2019, Table 1 |
| `eight_features` only: mean (SD) calcium, phosphate, bicarbonate, serum_albumin | The 4 additional 8-variable KFRE predictors | Tangri et al. 2016, *JAMA* supplement, eTable 1 |
| Mean / median follow-up, years (SD / IQR) | Standard cohort-study reporting | Major et al. 2019, Table 1; STROBE item 14(c) |
| Mean / median time-to-ESRD, years (SD / IQR), ESRD-positive subgroup | Same | Major et al. 2019, Table 1 |
| ESRD events (count) and incidence rate per 1,000 person-years (95% CI) | More informative than a raw percentage; standard epidemiological outcome reporting | Major et al. 2019, Table 1 |

Putting source cohort and final extracted cohort in the same table, side by side, is itself the check
for whether the merge selects a systematically different population (age, severity, outcome rate) —
no separate comparison section needed, per the same Table 1's own two-column structure (their cohort
vs. the UK-based CRIB/GLOMMS-1 cohorts).

Not included, and why: a standalone missing-data table (`four_features`/`eight_features` are already
filtered to complete cases by construction, same as Major et al.'s eligibility criteria — nothing new
to report); comorbidities such as cardiovascular disease/heart failure/hypertension/diabetes (this
repo's scenarios don't extract those fields at all, so they can't be reported).

The source-cohort baseline is `get_time_series_data_ckd_patients`'s actual population (CKD stage 3-5
diagnosis code **OR** ESRD diagnosis code, 34,332 patients) — not `get_ckd_patients_and_diagnoses
(late_stage=True)` (Task A's function), which filters CKD-3-5 codes only and undercounts to 10,179; see
the interim-finding note in EXPERIMENT_STATUS.md for how this was caught.

**Approval gate: do not proceed to full 1c (5-rep parallel extraction), nor implement the analysis
above, until the user has reviewed this section and approved.**

### 1c-0 result: are the extracted cohort sizes sufficient? (researched 2026-08-21)

The pilot's final `rep1` cohorts (patient-level, from the cohort-flow reports above) are **2,809
patients / 26,829 records** for `four_features` and **1,213 patients / 5,407 records** for
`eight_features`. Rather than judge this by feel, checked against the actual KFRE literature and a
standard statistical adequacy rule — conclusion: **sufficient, not undersized, by the standards of
this specific field.**

**1. The original derivation study itself** (Tangri et al. 2011, JAMA — the same paper whose
coefficients this repo uses) trained/validated on: development cohort 3,449 patients (386 events,
11%), validation cohort 4,942 patients (1,177 events, 24%). Our `four_features` (2,809) is ~80% the
size of that development cohort; `eight_features` (1,213) is about a third of it — smaller, but the
same order of magnitude as the study that produced the equation being benchmarked against.

**2. The 2016 multinational validation study's own per-cohort sample sizes for the 8-variable KFRE**
(eAppendix 1, section 1.5, of the same Tangri et al. 2016 JAMA supplement already used for the
coefficients — see the KFRE baseline model section above) — the most directly comparable data, since
it's the identical equation evaluated across 31 real-world cohorts of widely varying size:

| Cohort | N (8-variable analysis) |
|---|---|
| KPNW | 317 |
| CRIB | 263 |
| Geisinger | 414 |
| CCF ACR | 565 |
| MASTERPLAN | 568 |
| Mt Sinai BioMe | 625 |
| AASK | 898 |
| NephroTest | 1,205 |
| RENAAL | 1,409 |
| MDRD | 1,414 |
| Sunnybrook | 1,508 |
| SRR-CKD | 1,694 |
| CRIC | 2,896 |
| BC CKD | 10,917 |
| ICES-KDT | 12,955 |

9 of these 15 published, peer-reviewed cohorts are smaller than `eight_features`'s 1,213 (several far
smaller: 263–625), and `four_features`'s 2,809 lands mid-range, close to CRIC's 2,896. A published
KFRE validation study running on 263–625 patients is direct precedent that cohorts this size are
usable in this literature.

**3. An external KFRE validation study** (UK kidney transplant recipients, *BMC Nephrology* 2021, doi
10.1186/s12882-021-02259-4) used just 415 patients total — smaller than either of ours — and was
published without the N itself being treated as disqualifying.

**4. Events-per-variable (EPV) rule** (Peduzzi P, Concato J, Kemper E, Holford TR, Feinstein AR. "A
Simulation Study of the Number of Events per Variable in Logistic Regression Analysis." *J Clin
Epidemiol*. 1996;49(12):1373-1379 — ~7,900 citations, the standard reference for regression
sample-size adequacy): minimum 10 events per predictor variable for stable coefficient estimates.
Applying it with the patient-level ESRD-positive counts from the cohort-flow reports:
- `four_features`: 2,346 positive patients / 4 variables = **EPV ≈ 587** (59x the minimum)
- `eight_features`: 1,032 positive patients / 8 variables = **EPV ≈ 129** (13x the minimum)

Both comfortably clear EPV≥10.

**Caveat.** EPV≥10 is a classical-regression (logistic/Cox) heuristic; it directly validates the
closed-form KFRE benchmark (`kfre.py`) and the `cox.py` model, but has no established equivalent for
the more data-hungry models this repo also trains on the same data (`dynamic_deephit`,
`hazard_transformer`, `logistic_hazard`, `rnnsurv`) — more data is generically better for those than
the EPV floor implies. `eight_features` (1,213 patients) is the one to watch most closely once Stage
2's mini-experiment runs — not because it's unpublishable-small by KFRE-literature standards, but
because the neural models may be more N-sensitive than the closed-form benchmark.

### 1c. Data extraction (background, per rep, reusing existing scripts): parallel

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

## Stage 2.1: Feature-importance analysis
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
