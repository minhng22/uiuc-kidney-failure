# Paper Content-Gaps Remediation Plan ("to submit 2026")

**Last Updated:** 2026-09-18 (sunlab-serv-01.cs.illinois.edu) — plan written, nothing launched.

Addresses every **active** gap in
[paper/to submit 2026/README.md](paper/to%20submit%202026/README.md)'s "Known
content gaps (rejection risks)" section — items **3–13** (1 and 2 were closed
by the 2026-09-18 rep1–5 results landing). Lead paper is
`paper/to submit 2026/paper content/plos_digital_health.tex`; every text fix
must be hand-mirrored into `sn-article.tex` + `sections/*.tex` (shared by
`ml4h2026.tex`) per that README's "Which version is the live one".

No process has been launched from this plan. See "Background processes" below.

---

## 0. What was verified before writing this (evidence)

Every claim below was checked against the code/data on `sunlab-serv-01`, not
taken from the README or from comments.

| Claim | Evidence |
|---|---|
| All 5 reps shared **one** partition | `random_state=42` hardcoded, [model_data_store.py:63](pkgs/data_analysis/model_data_store.py#L63), `:197`, `:432`; corroborated by SD=0.0 across 5 reps for every deterministic model in `paper/to submit 2026/results/performance_summary.csv` |
| LogisticHazard trains against the test split | `val_data=(x_test, y_test)` at [logistic_hazard.py:82](pkgs/experiments/logistic_hazard.py#L82), built from `df_test` at `:48`/`:53` |
| IPCW reference is row-level, predictions are patient-level | `y_train = Surv.from_dataframe(..., data=df_train)` (raw multi-row frame) in `brier_score_up_to` [clinical_validity_analysis.py:419](pkgs/data_analysis/clinical_validity_analysis.py#L419) and in `discrimination_metrics`'s `cumulative_dynamic_auc` at `:474`; every model's `predictions()` returns one row per patient (e.g. [models/cox.py:21](pkgs/models/cox.py#L21) uses `get_last_observation_data`, [models/dynamicdeephit.py:293](pkgs/models/dynamicdeephit.py#L293) returns one curve per subject) |
| IBS scores a shared transform, not native curves | `brier_score_up_to(df_train, durations, events, risk_scores, horizon, baseline)` takes only the scalar risk score + Breslow baseline; `native_prob_fn` (4th element of every `predictions()` tuple) is threaded into `resolve_predicted_prob` for calibration/DCA but **never** into Brier |
| DCA comparators use a different unit than model curves | `treat_all_net_benefit_curve(df_test['duration_in_days'].values, ...)` and `egfr_threshold_net_benefit(egfr_referral_df...)` at [clinical_validity_analysis.py:722](pkgs/data_analysis/clinical_validity_analysis.py#L722)/`:729` are row-level; `model_net_benefit_curve(predicted, durations, events, ...)` at `:793` is patient-level |
| Twenty-feature DDH/HT importance OOM | `sample_size = len(X_test)` — whole test set in one backward pass, [feature_importance_analysis.py:525](pkgs/data_analysis/feature_importance_analysis.py#L525); test set is **1,613,452 rows × 40 cols**; report shows attempted allocations of 10,412,909,425,216 B (HT, mask path) and 77,445,696,000 B (DDH) — `generated_data/rep100/twenty_features_heterogeneous_shap_analysis_report.txt:107,151` |
| Prediction time is not a landmark | `pd.merge_asof(..., direction='nearest')` at [time_series_store.py:197](pkgs/data_analysis/time_series_store.py#L197); uACR matched whole-history unbounded (`tolerance=None`, `:366`), chemistry panel ±24h bidirectional (`:376`) |
| Demographics for subgroup analysis are cheap to obtain | `age`,`gender` already columns of `four_features`/`eight_features` CSVs; `race` joinable by `subject_id` via [store.py:33](pkgs/data_analysis/store.py#L33) `get_admission_df`; `patients.csv` (10 MB, present) carries `gender, anchor_age, anchor_year, anchor_year_group, dod` |
| Competing-risk blocker is a missing day-0 anchor, not missing `dod` | `duration_in_days = time - min(time) per subject` — [time_series_utils_store.py:24](pkgs/data_analysis/time_series_utils_store.py#L24); `dod` is present in `patients.csv`; only the per-subject anchor timestamp is absent from the exported CSVs |

### Three findings that change the plan

**(a) The paper's Cox numbers are from artifacts with a known, already-fixed
leak.** `cox.py` fit without naming covariates, so `duration_in_days` (the
outcome) and the unnamed CSV index column entered the design matrix —
[generated_data/rep100/cox_covariate_leakage_report.txt](generated_data/rep100/cox_covariate_leakage_report.txt).
The fix is in the code and verified on rep99 and rep100 (four_features
C-index 0.308 → 0.699), but reps 1–5 were produced **before** it. The paper
still reports Cox at 0.308/0.470/0.457 and
[discussion.tex:14](paper/to%20submit%202026/paper%20content/sections/discussion.tex) calls these
"below-chance and degenerate predictions … no causal explanation is
established here". That bullet is stale: the cause is established and fixed.
Retraining Cox is the single highest value-per-hour item in this plan.

**(b) rep1–rep5 do not exist on this host.** `generated_data/` holds only
`rep99` (122 MB), `rep100` (62 GB, = the former rep1) and `figs`. Nothing in
this plan that needs reps 1–5 can run here until they are rebuilt.

**(c) Another session is already rebuilding them, with a three-way split.**
[EXTERNAL_VALIDATION_REPS_EXPERIMENT_PLAN.md](EXTERNAL_VALIDATION_REPS_EXPERIMENT_PLAN.md)
— PID 1830030 on `sunlab-serv-02`, running
`pkgs.scripts.build_external_validation_reps`, 64/16/20 train/test/external
per rep, **own seed per rep (42+rep)**, stratified on the patient ESRD label,
sourced from rep100 (no re-extraction). Loaders
(`get_train_test_external_data`) and `commons.py` paths are already in.
That is not this session's row; do not touch it. It supplies the data
substrate for W2/W3/W5 below, so this plan is **gated on it finishing**.

### Runtime budget (the binding constraint)

Measured from rep4's per-model log mtimes (launch Aug 29 19:02, 11 models in
parallel on one host — mtimes, so upper bounds on true runtime):

| Model | Wall clock |
|---|---|
| weibul, kfre, survival_svm, deepsurv | minutes – 1 h |
| logistic_hazard, cox | ~3 h |
| srf | ~6 h |
| hazard_transformer, rnnsurv | ~1.5 days |
| gbsa | ~13 days |
| dynamic_deephit | ~15 days |

**A full 5-rep × 3-scenario retrain is a ~2-week wall-clock job**, dominated by
GBSA and Dynamic-DeepHit on `twenty_features_heterogeneous`. This is why the
plan batches *every* code fix into **one** retrain (W2) rather than fixing
gaps one at a time. 4 of 8 GPUs on `sunlab-serv-01` are free (RTX 2080 Ti,
11 GB each); the two long poles are CPU-bound.

---

## 1. Strategy

Gaps 3, 4, 5 and 7 all invalidate the same artifacts, so they get fixed
**before** a single retrain, not after. Gaps 6, 9, 12, 13 are
text/analysis-only and run in parallel with it. Gaps 8 and 10 need a change to
the extraction pipeline and are the only two that need a decision from the
user before starting (§4).

```
W1 pipeline fixes  ──┐
                     ├─► W2 ONE retrain (5 reps × 3 scenarios, 3-way split) ─► W6 CI/tests ─► W8 paper sync
(serv-02 rep build) ─┘                                                    └─► W3 external eval
W4 importance fix ───────────────────────────────────────────────────────────────────────────┘
W5 subgroup/fairness ────────────────────────────────────────────────────────────────────────┘
W7 text + checklist + ethics (independent, start any time) ──────────────────────────────────┘
```

---

## 2. Workstreams

### W1 — Pipeline correctness fixes (gaps 4, 5; unblocks everything)

All code-only. Per [CLAUDE.md](CLAUDE.md), each is verified on **rep99**
(`python -m pkgs.scripts.run_experiments analyze --reps 99 …`) before any
full-scale rep is launched.

| # | Gap | Fix | Files |
|---|---|---|---|
| W1.1 | 4 — LogisticHazard trains on the test split | Carve a validation partition out of **train** (patient-level, same stratification as the rep builder) and pass that as `val_data`; use it for Optuna trial selection and checkpointing. The 3-way reps make this clean: tune on a slice of the 64% train, never touch the 16% test. | [pkgs/experiments/logistic_hazard.py:48-82](pkgs/experiments/logistic_hazard.py#L48), `:195-217` |
| W1.2 | 5a — IPCW censoring reference is row-level | Build `y_train` for IBS **and** `cumulative_dynamic_auc` from the one-row-per-patient training frame (`get_last_observation_data(scenario, split='train')`), matching the unit of `durations`/`events`. Assert `len(y_train)` equals the patient count, so a future regression fails loudly. | `brier_score_up_to` [clinical_validity_analysis.py:390-431](pkgs/data_analysis/clinical_validity_analysis.py#L390), `discrimination_metrics` `:451-489` |
| W1.3 | 5b — IBS scores a shared scalar transform | Thread `native_prob_fn` (already the 4th element of every `predictions()` tuple) into the Brier path; use each model's native survival curve where it has one, fall back to the Breslow baseline only where it does not, and **label which was used per model** in the report and the paper table. | same two functions + `resolve_predicted_prob` `:437` |
| W1.4 | 5c — DCA comparators use a different unit | Compute treat-all / eGFR-threshold net benefit from the same patient-level frame the model curves use (`get_last_observation_data`), not `df_test` rows. This is what withdrew the net-benefit claims; fixing it **restores the paper's clearest actionable finding**. | [clinical_validity_analysis.py:722,729](pkgs/data_analysis/clinical_validity_analysis.py#L722) |
| W1.5 | (a) above — Cox leak | Already fixed in `cox.py`; confirm the fix is present, and let W2 regenerate the artifacts. No separate run. | [pkgs/experiments/cox.py](pkgs/experiments/cox.py) |

**Exit criterion:** rep99 clinical-validity + feature-importance reports
regenerate with no new errors, IBS/AUC/DCA numbers move in an explainable
direction, and each moved number is written up in
`generated_data/rep99/w1_pipeline_fix_report.txt`.

### W2 — The single retrain (gaps 3, 4, 5, 7 substrate)

Gated on: W1 merged + rep99-verified, **and** `sunlab-serv-02`'s PID 1830030
finishing all three scenarios for rep1–5.

- Train all 11 models × 3 scenarios × reps 1–5 on the new **64% train** split,
  using `pkgs/scripts/run_rep.sh` / `run_experiments train`. Per CLAUDE.md,
  read each `__main__` block first, or drive
  training through `run_experiments.py train --models … --scenarios …`, which
  calls each model's run function directly and bypasses its `__main__` block —
  those blocks have hardcoded scenario lists that have caused an off-scope
  2-hour raw extraction before. (CLAUDE.md points at
  `pkgs/scripts/run_stage2_new_scenarios.py` as the scoped-driver pattern;
  that file is **not present** in this tree.)
- Split across `sunlab-serv-01/02/03`; record PID + host + log per rep in
  §5 below.
- **Scope decision to make first (§4, D1):** at ~2 weeks, consider capping
  `twenty_features_heterogeneous` GBSA/DDH (patient subsample or fixed trial
  budget) and reporting the cap, rather than serialising the paper behind it.
- Re-aggregate with `paper/to submit 2026/scripts/aggregate_results.py` into
  `results/performance_{summary,per_run}.csv` + `performance_provenance.json`.

**What this buys in the paper:** reps become genuinely independent patient
partitions, so the reported SD stops being "5 refits on one partition" and
the Limitations bullet about a fixed partition can be dropped; Cox's
below-chance numbers disappear; LogisticHazard becomes a real held-out
estimate.

### W3 — Held-out evaluation + honest framing (gap 7)

- Evaluate every trained model on each rep's `external_validation` split via
  `get_train_test_external_data`, producing
  `generated_data/rep<N>/<scenario>_external_validation_report.txt` and a
  second results table (`results/external_validation_summary.csv`).
- **Naming caution — state this in Methods:** a random 20% of the *same*
  MIMIC-IV cohort is a second internal split, not external validation in the
  TRIPOD sense. A reviewer at a soundness-focused venue will say so. Two
  options, both cheap, and I recommend doing the first:
  - **Temporal validation (recommended).** `patients.csv` carries
    `anchor_year_group` (e.g. 2008–2010 … 2017–2019). Split the pool by
    year group — train on earlier, validate on later. That is TRIPOD type 2b
    "narrow external", needs only a `subject_id` join, and is a genuinely
    defensible claim.
  - Keep the random split, and call it "internal–external / held-out", never
    "external validation".
- A true independent-source cohort (eICU-CRD, MIMIC-III) is the only thing
  that fully closes gap 7. Out of scope here unless the user wants it (§4, D4).

### W4 — Memory-bounded feature importance (gap 11)

Independent of W2 — runs today against rep100's existing twenty-feature
artifacts.

- Replace the single whole-test-set backward pass with chunked accumulation:
  iterate mini-batches, accumulate `|grad|` sums and a row count, divide at
  the end. Identical result, bounded memory. Add a `--max-rows` / sampling cap
  as a second guard.
- The Hazard Transformer path also builds a `batch_size × seq_len` mask over
  1.6 M rows (the 10.4 TB allocation) — chunk that with the batch, and build
  the mask per chunk.
- Verify on rep100 twenty-feature DDH + HT, then rerun for reps 1–5 after W2.
- Re-cite the resulting
  `twenty_features_heterogeneous_all_models_feature_importance.png` in the
  paper (README notes it is currently uncited), and drop the
  "failed memory allocation in all five runs" limitation.

Files: [feature_importance_analysis.py:523-560](pkgs/data_analysis/feature_importance_analysis.py#L523),
`extract_gradient_based_importance` `:538-921`.

### W5 — Subgroup / fairness analysis (gap 12)

New analyzer `pkgs/data_analysis/subgroup_analysis.py`, wired into
`run_experiments.py --analyses subgroup` so it inherits the existing
rep/scenario/model selection and logging.

- Strata: **sex** (`gender`, in-file), **age band** (`age`, in-file; <50 /
  50–64 / 65–74 / 75+), **race** (join `get_admission_df` by `subject_id`,
  collapse to the census-style groups `demographics.py` already produces).
  `twenty_features_heterogeneous` lacks `age`/`gender` columns — join from
  `patients.csv` (`anchor_age`, `gender`) on `subject_id`.
- Per stratum, per model: n, event rate, C-index, IBS, calibration
  intercept/slope at 2 yr, plus the gap vs. the overall cohort.
- Output `generated_data/rep<N>/<scenario>_subgroup_report.txt` + a forest-style
  PNG; report in the paper as a supplementary table with the smallest-stratum
  n stated (eight_features has only 1,213 patients pooled, so some strata will
  be too small to interpret — say so rather than reporting a noisy number).

### W6 — Uncertainty quantification (gap 3, the part W2 does not close)

Even with 5 independent partitions, the paper still has no intervals or tests.

- **Patient-level bootstrap.** B = 1000 resamples with replacement of test-set
  *patients* (not rows), recomputing C-index / IBS / AUC per model per
  resample; report percentile 95% CIs. Predictions are computed once and
  re-indexed per resample, so this costs minutes, not a retrain.
- **Paired comparisons.** On the same bootstrap resamples, compute the paired
  difference distribution for each model vs. KFRE (the paper's reference
  comparator) and vs. the best model; report the difference CI and the
  proportion favouring each side. Paired-on-resample avoids the independence
  assumption a naive test would need.
- Combine across the 5 reps by reporting per-rep CIs plus the across-rep
  range — do not pool resamples across different partitions.
- New `pkgs/data_analysis/bootstrap_ci.py`, exposed as
  `run_experiments.py --analyses bootstrap`; writes
  `results/performance_ci.csv`.
- Paper tables change from `mean (SD)` to `mean [95% CI]`, and the Limitations
  bullet about "no bootstrap confidence intervals, paired significance tests"
  is deleted.

### W7 — Text, reporting-guideline and ethics fixes (gaps 6, 9, 13; no compute)

Can start immediately, in parallel with everything above.

- **Gap 6 — spectrum bias.** [methods.tex:26](paper/to%20submit%202026/paper%20content/sections/methods.tex#L26)
  currently frames the 83–92% ESRD rate only as "event-rich by construction".
  Add, in Methods (not just Limitations): KFRE was derived and validated in
  **outpatient nephrology-referral** CKD populations with an ESRD incidence
  one to two orders of magnitude lower; an ICU/hospital cohort selected on a
  CKD/ESRD diagnosis code is a different point on the disease spectrum, which
  biases both KFRE's calibration (predicted risks far below observed) and
  every discrimination comparison against it. Mirror into
  `plos_digital_health.tex:148` and `sections/ml4h_methods.tex`.
- **Gap 9 — TRIPOD.** Complete a **TRIPOD+AI** (2024) checklist — the current
  STROBE citation covers cohort flow only and is the wrong instrument for a
  multivariable prediction-model study. Deliverable:
  `paper/to submit 2026/paper content/tripod_ai_checklist.md` (+ a PDF/S-file
  for submission), a one-line Methods statement citing it, and a
  `\bibitem` for the TRIPOD+AI statement in `sn-bibliography.bib`. Doing this
  *before* the rewrite is deliberate: the checklist will surface any remaining
  unreported item while the text is still being edited.
- **Gap 13 — ethics.** [sn-article.tex:73-75](paper/to%20submit%202026/paper%20content/sn-article.tex#L73)'s
  declaration is generic. Add the language PLOS expects for MIMIC
  submissions: PhysioNet credentialed-user status, completion of the CITI
  "Data or Specimens Only Research" course, acceptance of the PhysioNet
  Credentialed Health Data Use Agreement, the BIDMC/MIT IRB waiver of informed
  consent under which MIMIC-IV is released, and the MIMIC-IV version + DOI.
  Uncomment and fill the corresponding block at
  `plos_digital_health.tex:411-417`.
- **Bibliography hygiene** (README flags these as latent, not introduced):
  missing `journal` on `ishwaran2008random`, missing `author`/`publisher` on
  `hu2022locf_bias`, conflicting `volume`/`number` on `lee2018deephit` in
  `sn-bibliography.bib`. Fix once, before submission; it is shared by all
  three venue files.

### W8 — Paper synchronisation and rebuild (all gaps land here)

- Rewrite Results/Discussion around the new numbers; delete every limitation
  the work above closes (fixed partition, no CIs, LogisticHazard leakage,
  row-level IPCW, transformed-score IBS, withdrawn DCA claims, failed
  twenty-feature importance, no subgroup analysis).
- Re-copy PNGs from `generated_data/rep<N>/` into `paper content/figs/`,
  and cite the two `twenty_features_heterogeneous_*` figures that are
  currently uncited.
- Edit the lead file (`plos_digital_health.tex`), then **hand-mirror into
  `sections/*.tex`** (→ `sn-article.tex`, `ml4h2026.tex`); PLOS duplicates
  prose rather than `\input`-ing it, and this has already desynced once.
  `ml4h2026.tex` is double-blind — grep for author/institution/repo strings
  before touching it.
- Recompile all three with tectonic; drop timestamped PDFs into `drafts/`;
  clean build litter.

---

## 3. Gap → workstream coverage

| Gap | Severity (README) | Workstream | Compute | Fully closed? |
|---|---|---|---|---|
| 3 — no patient-sampling uncertainty | near-certain reject | W2 (independent partitions) + **W6** (bootstrap CIs, paired tests) | minutes after W2 | Yes, except external cohort |
| 4 — LogisticHazard tunes on test | near-certain reject | **W1.1** + W2 | in W2 | Yes |
| 5 — IPCW unit, IBS transform, DCA unit | near-certain reject | **W1.2/1.3/1.4** | rep99 minutes; reruns in W2 | Yes — and restores the net-benefit finding |
| 6 — population mismatch vs KFRE domain | high | **W7** | none | Yes (framing) |
| 7 — no external cohort | high | **W3** (held-out + temporal), D4 for a true second source | eval only | Partly — see D4 |
| 8 — not a prospective landmark | high | **W9** (below), needs D2 | ~2 h/scenario re-extraction + full retrain | Only if D2 = yes |
| 9 — no TRIPOD checklist | high | **W7** | none | Yes |
| 10 — competing risks | moderate | **W10** (below), needs D3 | 30–60 min anchor scan + analysis | Depends on D3 |
| 11 — twenty-feature importance OOM | moderate | **W4** | hours | Yes |
| 12 — no subgroup/fairness analysis | moderate | **W5** | hours | Yes |
| 13 — ethics declaration | moderate | **W7** | none | Yes |

### W9 — Prospective landmark design (gap 8; conditional on D2)

The honest fix, if approved: pick a landmark time `t0` per patient (first
eligible creatinine draw, or a fixed calendar offset), keep only measurements
with `charttime <= t0`, reset `duration_in_days` to `t - t0`, and drop
patients whose event precedes `t0`. Concretely:
`merge_asof(direction='nearest')` → `direction='backward'` with an explicit
tolerance at [time_series_store.py:197](pkgs/data_analysis/time_series_store.py#L197),
so the unbounded whole-history uACR match (`:366`) and the ±24 h bidirectional
chemistry match (`:376`) can no longer pull in post-anchor measurements.

Cost: changes the extracted data, so it means re-extraction from
`labevents.csv` (~2 h/scenario) **plus a second full retrain** (~2 weeks). It
is the single most expensive item in this plan. Cheaper alternative, if D2 is
"no": quantify the exposure instead — report what fraction of matched uACR
values post-date their creatinine anchor and by how many days, and cite that
number in Limitations. That turns an open-ended objection into a bounded,
measured one for a few hours' work.

### W10 — Competing risks (gap 10; conditional on D3)

The 2026-08-23 decision to skip this was based on the exported CSVs lacking
each subject's absolute day-0 anchor. That is accurate but the blocker is
smaller than recorded: `dod` is present in `patients.csv`, and the anchor is
just `min(charttime)` per subject over the same filtered lab frame
([time_series_utils_store.py:24](pkgs/data_analysis/time_series_utils_store.py#L24)).
A scoped one-pass script (`pkgs/scripts/build_subject_anchor_dates.py`)
streaming `subject_id, charttime, itemid` out of `labevents.csv` with the same
per-scenario itemid filter yields `subject_id → anchor_date` in one ~30–60 min
scan — no pipeline rewrite, no retrain. Then:

- Fine–Gray subdistribution model and/or cause-specific cumulative incidence
  for ESRD with death as a competing event, reported as a sensitivity analysis
  against the current cause-specific results.
- If the CIF materially differs from `1 − KM`, the paper's absolute-risk
  and calibration claims need requalifying — which is exactly the reviewer
  objection this gap anticipates.

---

## 4. Decisions needed before anything launches

| # | Decision | Why it can't be defaulted | My recommendation |
|---|---|---|---|
| **D1** | Cap `twenty_features_heterogeneous` GBSA/DDH for the W2 retrain (subsample patients / fix the trial budget), or run them to completion? | Uncapped, W2 is ~2 weeks and everything downstream waits on it. Capping is defensible if reported, but it is a scope change to published results. | Cap, and report the cap. Four/eight-feature scenarios finish in hours either way, so the paper can be drafted against those while twenty-feature completes. |
| **D2** | Do the landmark re-extraction (gap 8)? | ~2 h/scenario + a second ~2-week retrain, and it changes the cohort. | Not for this submission. Do the measured-exposure alternative in W9 instead, and name the landmark redesign as the planned next study. |
| **D3** | Reopen competing risks (gap 10)? The earlier "not planned" decision assumed a much larger cost than the anchor-scan route actually needs. | It reverses a recorded user decision. | Reopen — ~1 day total, and it converts a likely reviewer-mandated revision into a completed sensitivity analysis. |
| **D4** | Pursue a true second-source external cohort (eICU-CRD / MIMIC-III)? | Weeks of new extraction work; the only thing that fully closes gap 7. | No for this submission. Ship W3's temporal validation and state the limitation precisely. |
| **D5** | Is `sunlab-serv-02`'s three-way rep build the agreed substrate for the paper's final numbers? | It is another session's run; W2 depends on it, and it changes the split rule behind every published number. | Confirm with that session/the user before W2 launches. |

---

## 5. Background processes

*(none launched from this plan)*

| PID | Host | Command | Log | Started | Status | Blocking / shared state |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

**Depends on, but does not own:** PID 1830030 on `sunlab-serv-02`
(`build_external_validation_reps`) — tracked in
[EXTERNAL_VALIDATION_REPS_EXPERIMENT_PLAN.md](EXTERNAL_VALIDATION_REPS_EXPERIMENT_PLAN.md).
Not this session's row; do not edit or restart it.

**Shared state this plan will touch when it runs:** `CKD_REP`/`current_rep`
([pkgs/commons.py:155](pkgs/commons.py#L155)) per worker process, and
`generated_data/rep1`–`rep5` (currently being written by the serv-02 build —
W2 must not start until that finishes).

## 6. Status

| Workstream | Status | Notes |
|---|---|---|
| W1 pipeline fixes (gaps 4, 5) | not started | rep99 verification required before W2 |
| W2 single retrain (gaps 3, 4, 5, 7) | blocked | on W1 + serv-02 rep build + D1/D5 |
| W3 held-out + temporal validation (gap 7) | not started | eval-only, follows W2 |
| W4 chunked importance (gap 11) | not started | can start now against rep100 |
| W5 subgroup/fairness (gap 12) | not started | can start now against rep100 |
| W6 bootstrap CIs + paired tests (gap 3) | not started | follows W2 |
| W7 text/TRIPOD/ethics (gaps 6, 9, 13) | not started | no compute; can start now |
| W8 paper sync + rebuild | not started | last |
| W9 landmark (gap 8) | awaiting D2 | measured-exposure fallback can start now |
| W10 competing risks (gap 10) | awaiting D3 | anchor-scan route, ~1 day |
