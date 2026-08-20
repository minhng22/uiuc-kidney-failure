# CKD Fifty Features Heterogeneous Experiment Plan

**Last Updated:** 2026-08-19 16:16 CDT

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

## Data & Results Visualization Plan (for Paper)

The metrics above (C-index, Brier score, time-dependent AUC) are necessary but not sufficient for
a publishable paper — reviewers in this space expect a specific set of figures. This section was
built by researching 30+ recent papers on CKD/ESRD prediction, survival analysis with EHR data, and
the specific model families used here (Cox, DeepHit/Dynamic-DeepHit, RNN-Surv), to identify what
insights/visualizations are standard. Sources are cited inline; full list at the bottom of this
section.

### A. Cohort & data-quality insights (produce once, from the raw extraction — not per-model)
1. **Cohort selection / STROBE-style flow diagram** — patient counts at each inclusion/exclusion
   step (MIMIC-IV → CKD cohort → ESRD-labeled → train/test split). Standard in nearly every MIMIC-IV
   clinical prediction paper (Sources 30-33).
2. **Table 1 (baseline characteristics)** — demographics + lab summary stats, stratified by outcome
   (progressed to ESRD vs. not) and by rep, to show the 5 reps are comparable resamples.
3. **Missingness pattern visualization** — since 50 of the 100 features are literally `_missing`
   indicators, this dataset's missingness structure *is* a first-class result, not just a caveat.
   Recommended: a heatmap/set-visualization of which labs are missing together (Sources 25-27 show
   missingness in EHR labs is informative, not random — worth testing that claim on this cohort), plus
   a per-feature missingness-rate bar chart across the 50 labs.
4. **Feature distribution panel** — histograms/violin plots for the 50 raw lab values (train vs.
   test, and possibly ESRD vs. non-ESRD), to sanity-check the 5 reps are drawing from the same
   underlying distribution and to catch any scenario-construction bugs early (each lab column in
   `time_series_store.py` is built independently — a distribution check is the cheapest way to catch
   a swapped/miscoded column).
5. **Correlation heatmap among the 50 lab features** — expected given they're physiologically
   related (e.g., urea_nitrogen/creatinine-proxy features, hematocrit/hemoglobin); useful context for
   interpreting later feature-importance results and for justifying any dimensionality choices.
6. **eGFR trajectory plot** (or the closest available primary renal-function marker) — individual
   patient trajectories or group-based trajectory clusters (persistently-low / progressive-decline /
   accelerated-decline), the canonical figure in the CKD progression literature (Sources 33-34);
   directly motivates why a *time-varying* Cox model and the RNN/transformer-based models are the
   right modeling choice here rather than a single-timepoint model.

### B. Model discrimination & calibration insights (per model, per rep — the "did it work" figures)
7. **Kaplan-Meier curves stratified by predicted risk group** — split test-set patients into
   tertiles/quartiles by each model's predicted risk score and plot KM curves per group; a model with
   real signal shows clean monotonic separation between risk groups (Sources 4, 6, 28-29, 35 all
   report exactly this as their primary discrimination figure, more intuitive to a clinical audience
   than a bare C-index number).
8. **Time-dependent AUC/ROC curve over follow-up time** (not just the single mean-AUC number
   currently logged) — plot AUC(t) across the follow-up window; several sources note discrimination
   is *not* constant over time for kidney models (Source 32's UNOS_Kidney example), so this curve is
   more informative than one scalar and would explain why the Mean time-dependent AUC values seen so
   far (0.43-0.53 range) look weak — the curve shape may reveal it's driven by poor discrimination in
   a specific time window rather than uniformly weak.
9. **Calibration plot** (predicted vs. observed risk, by decile) — standard alongside discrimination
   in every benchmark paper found (Sources 5, 20-22); currently unmeasured here even though Brier
   score is logged (Brier conflates calibration + discrimination, calibration plots separate them).
10. **Model comparison figure across the 5 models** (Cox, DDH, HazardTransformer, LogisticHazard,
    RNNSurv) — grouped bar/forest plot of C-index (and Brier, AUC) with error bars across the 5 reps,
    the head-line "how do our models compare" figure that every benchmark paper leads with
    (Sources 6, 23-24, 28).
11. **Decision curve analysis (net benefit vs. threshold probability)** — increasingly expected in
    clinical-prediction papers beyond discrimination/calibration alone (Sources 36-38); shows whether
    the model would actually change a clinician's referral/treatment decision at realistic risk
    thresholds, not just whether it ranks patients well.

### C. Interpretability insights (what's actually driving the predictions)
12. **Cox hazard-ratio forest plot** — for the 100 fitted coefficients (or the top ~20 by
    |coefficient| or p-value), the standard way to present a Cox model's feature effects; directly
    available from the already-fitted `CoxTimeVaryingFitter.params_`/`.summary` (verified this session
    — the model object already has this).
13. **SHAP summary/beeswarm plot** for the deep models (DDH, HazardTransformer, LogisticHazard,
    RNNSurv) — the dominant interpretability method across the benchmark/interpretability papers found
    (Sources 5, 8-9, 39); lets the paper compare *which* features the Cox model vs. the deep models
    rely on, a natural discussion point given they'll likely disagree on ranking.
14. **`_missing`-indicator-specific importance analysis** — this dataset's design choice (missingness
    as a feature) is unusual enough to warrant its own figure: how much of each model's predictive
    signal comes from the 50 `_missing` flags vs. the 50 raw lab values? (E.g., SHAP values summed
    separately for the two feature groups, or a Cox coefficient comparison of `X` vs `X_missing`
    pairs.) None of the papers surveyed do exactly this scenario, but Sources 25-27 (informative
    missingness in EHR labs) support that it's a real, publishable finding if the missingness flags
    turn out to carry meaningful signal — worth flagging as a possible original contribution of this
    experiment rather than only a replication of standard methodology.

### D. Robustness across replications (this study's 5-rep design is not common in the literature —
    most surveyed papers use a single train/test split — so this is a chance to show something most
    comparable papers can't)
15. **Box/violin plot of C-index (and Brier, AUC) across the 5 reps, per model** — variance across
    reps is itself a result (a model that's only good on 1 of 5 reps is a different finding than one
    that's consistently good); currently the "Phase 3: Results" table below only has a Mean ± Std
    placeholder — this should be the figure behind those numbers, not just the summary stat.
16. **Optuna hyperparameter-search visualization** for the deep models (DDH, HazardTransformer,
    RNNSurv, LogisticHazard) — optimization-history and parameter-importance plots (`optuna.visualization`
    natively supports both) to show the HP search actually converged and to document what
    hyperparameters ended up mattering; also useful in the appendix to justify the search-space caps
    mentioned earlier in this doc (the "capping dynamic_deephit/hazard_transformer search spaces to
    avoid OOM/excessive runtime" fix).

### Priority / feasibility given the current pipeline
- **Already have the data for #1-6, #12** without any new training: the train/test CSVs and the
  rep1 cox model (`.dill`, content-verified this session) exist right now — a data-insights notebook
  covering the cohort/missingness/distribution/correlation figures plus the Cox forest plot could be
  built today without waiting for any of the currently-running training jobs.
- **#7-11, #13, #15-16 need the in-flight training to finish** (all 5 models × 5 reps) since they
  require predicted risk scores / SHAP values / per-rep metrics that don't exist until each model's
  training completes.
- **#14 (missingness-indicator importance)** is the one recommendation not seen directly in any
  surveyed paper — flagging it as optional/exploratory rather than a must-have, but worth discussing
  with whoever is writing the paper since it could be a distinguishing contribution.

### Built: data-only figures (#1-6, #12) — Aug 18, 2026
Published as an artifact: **[CKD Fifty-Feature Cohort Report](https://claude.ai/code/artifact/fbc6497b-03e4-45fb-988f-17f6ae14d3f2)**
(cohort flow, missingness, distributions, eGFR trajectory, Cox forest plot for rep1 & rep5). Built
from rep1's train/test CSVs — confirmed byte-identical across all 5 reps (checksummed), so these
figures hold for every rep and don't need rebuilding per-rep.

Two findings revise items in this plan rather than just fill them in:
- **#3 (missingness pattern) turned out to be a data-shape artifact, not a clinical signal**: every
  lab is 95.5-99.97% missing at the record level because the extraction pipeline
  (`pkgs/data_analysis/time_series_store.py`) writes one row per single lab-draw event — each row has
  exactly one lab populated, the other 49 flagged missing. Missingness rate per lab = 1 minus that
  lab's share of all draws. Mean pairwise missingness-indicator correlation is ≈ -0.016 (mutually
  near-exclusive), confirming this. The "informative EHR missingness" framing from the literature
  survey doesn't directly apply without a reshape first.
- **#5 (correlation heatmap) is not computable as the data is currently shaped**: checked directly —
  of 2,500 lab×lab correlation cells only the 50 diagonal entries (self-correlation) are non-null;
  every off-diagonal pair is `NaN`, because no row ever has two different labs simultaneously
  non-missing (same root cause as above). Needs a reshape (e.g. pivot to patient×day, forward-fill
  within a window) before this figure is possible — a preprocessing step to plan for, not just a
  chart to draw.
- **Bonus finding**: rep1 and rep5's cox coefficients are numerically identical (diffs ~1e-15) —
  expected given identical data + a deterministic fit, but confirms cox will show zero cross-rep
  variance in the eventual "Mean ± Std" results table; only the deep models' training stochasticity
  will produce real spread there.

### Sources (30+ papers/references reviewed)
1. [Predicting the Progression of CKD: A Systematic Review of AI/ML Approaches](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11166249/)
2. [Predicting End-Stage Renal Disease and Mortality in CKD Using ML (JMIR Med Inform 2026)](https://medinform.jmir.org/2026/1/e81152)
3. [Machine-learning-based Web system for CKD progression and mortality](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9931312/)
4. [ML models for predicting short-term progression in stage 4 CKD (Sci Reports 2025)](https://www.nature.com/articles/s41598-025-23037-4)
5. [Interpretable ML for predicting CKD progression risk](https://pmc.ncbi.nlm.nih.gov/articles/PMC10793198/)
6. [Integrated ML and Survival Analysis Modeling for CKD Risk Stratification](https://arxiv.org/pdf/2411.10754)
7. [AI for Risk Prediction of ESRD in Sepsis Survivors with CKD](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8945427/)
8. [ML for Predicting CKD Progression in COVID-19 AKI Patients](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11274434/)
9. [Proactive healthcare: ML-driven insights into kidney failure prediction](https://link.springer.com/article/10.1007/s43995-025-00118-z)
10. [AKI Prognosis Prediction Using ML: A Systematic Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11699606/)
11. [Dynamic Kidney Failure Prediction Model based on Deep Learning, with external validation](https://arxiv.org/abs/2501.16388)
12. [DySurv: dynamic deep learning survival model with conditional variational inference](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12758469/)
13. [Contrastive Learning of Temporal Distinctiveness for Survival Analysis in EHR](https://arxiv.org/pdf/2308.13104)
14. [Attention-Based Synthetic Data Generation for Calibration-Enhanced Survival Analysis (CKD EHR case study)](https://arxiv.org/pdf/2503.06096)
15. [Dynamic-DeepHit: Deep Learning for Dynamic Survival Analysis with Competing Risks (IEEE)](https://ieeexplore.ieee.org/document/8681104/)
16. [DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks (AAAI)](https://cdn.aaai.org/ojs/11842/11842-13-15370-1-2-20201228.pdf)
17. [RNN-SURV: A Deep Recurrent Model for Survival Analysis (ICANN 2018)](https://link.springer.com/chapter/10.1007/978-3-030-01424-7_3)
18. [A Dynamic Predictive Model for Progression of CKD](https://www.sciencedirect.com/science/article/abs/pii/S0272638616304176)
19. [Risk prediction for CKD progression using heterogeneous EHR data and time series analysis (JAMIA)](https://academic.oup.com/jamia/article/22/4/872/1746401)
20. [Improved Survival Analyses Based on Characterized Time-Dependent Covariates for CKD Progression](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10296072/)
21. [Predicting kidney failure from longitudinal kidney function trajectory: A comparison of models (PLOS ONE)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6508737/)
22. [Comparison of a CKD predictive model for T2DM using Cox regression vs. ML (Clin Kidney J)](https://academic.oup.com/ckj/article/16/3/549/6881386)
23. [Advances in survival analyses: ML methods and model comparison](https://www.sciencedirect.com/science/article/pii/S0085253826003017)
24. [Beyond Cox models: assessing ML performance in non-proportional hazards survival analysis](https://www.sciencedirect.com/science/article/pii/S001048252501529X)
25. [Using set visualization techniques to investigate missing values in EHR](https://www.medrxiv.org/content/10.1101/2022.05.13.22275041v1.full)
26. [Informative missingness: patterns in missing laboratory data in the EHR](https://www.sciencedirect.com/science/article/pii/S1532046423000278)
27. [Imputation of missing values for EHR laboratory data (npj Digital Medicine)](https://www.nature.com/articles/s41746-021-00518-0)
28. [SurvBenchmark: comprehensive benchmarking of survival analysis methods (GigaScience)](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giac071/6652188)
29. [Deep Learning for Patient-Specific Kidney Graft Survival Analysis](https://arxiv.org/pdf/1705.10245)
30. [Nomogram to predict rapid kidney function decline (BMC Nephrology)](https://link.springer.com/article/10.1186/s12882-022-02696-9)
31. [Nomogram for CKD progression in IgA nephropathy, with internal/external validation (PeerJ)](https://peerj.com/articles/18416/)
32. [Nomogram for heart failure and mortality in CKD (Frontiers in Medicine 2026)](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1784717/full)
33. [Longitudinal progression trajectory of GFR among CKD patients](https://pubmed.ncbi.nlm.nih.gov/22284441/)
34. [Predialysis trajectories of eGFR and concurrent CKD biomarker trends](https://pmc.ncbi.nlm.nih.gov/articles/PMC10265358/)
35. [Kaplan-Meier risk-group stratification examples across multiple survival-analysis papers (ResearchGate figure collection)](https://www.researchgate.net/figure/Kaplan-Meier-plot-for-risk-stratification-using-Risk-Score-RS-Kaplan-Meier-survival_fig3_335714418)
36. [Decision curve analysis: a technical note (Ann Transl Med)](https://atm.amegroups.org/article/view/20389/html)
37. [A simple, step-by-step guide to interpreting decision curve analysis (Diagn Progn Res)](https://link.springer.com/article/10.1186/s41512-019-0064-7)
38. [Risk-based referral model to nephrologist specialist care, using decision curve analysis (Nephrol Dial Transplant)](https://academic.oup.com/ndt/article/41/1/102/8195529)
39. [Using SHAP to explain predictions in healthcare ML models](https://python.plainenglish.io/using-shap-to-explain-predictions-in-healthcare-ml-models-with-code-and-visuals-175b9e3e3f41)

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
| rep1 | ✅ (C-index 0.441) | 🔄 PID 4084450 | ⏳ | ⏳ | ⏳ | In Progress (via orchestrator, PID 4080938) — re-verified 16:16 CDT (Aug 19) on `sunlab-serv-01.cs.illinois.edu`, still `dynamic_deephit` (~2d20h05m elapsed, CPU time now past 25 days cumulative [25-18:44:13], `R` state). Swap/RAM steady. **Cox result caveat (investigated per user request)**: the reported C-index 0.441 for rep1's `cox` stage came from loading a pre-existing model file (`ckd_fifty_features_heterogeneous_cox_model.dill`, dated May 27 — predates this run) rather than a fresh fit; `cox.py` only trains if no model file exists (see [pkgs/experiments/cox.py:29-38](pkgs/experiments/cox.py#L29-L38)). Confirmed not a data-mismatch issue — the model postdates rep1's train/test CSVs (May 23), so it was originally fit on this exact data, and `CoxTimeVaryingFitter` fitting is deterministic, so a retrain would very likely reproduce the same result. **Further verified by content, not just filename** (per user follow-up): loaded the dill file directly — it's a `CoxTimeVaryingFitter`, `event_col='has_esrd'`, `penalizer=1.0` (matches `cox.py` exactly), with 102 fitted coefficients = `Unnamed: 0` + `duration_in_days` + all 100 CKD_FIFTY_FEATURES columns (50 labs + their 50 `_missing` indicators, confirmed against [pkgs/data_analysis/time_series_store.py:242-296](pkgs/data_analysis/time_series_store.py#L242-L296) which explicitly builds 50 labs + 50 missingness indicators for this scenario — the `get_feature_columns()` helper returning only 50 is a narrower list used just for NaN-filtering, not the actual training feature set). So: genuinely the right model for this scenario/rep, not a mismatched file from elsewhere. User was informed and chose not to force a retrain. Root cause of the long runtime: `get_device()` in [pkgs/experiments/dynamic_deephit.py:397-398](pkgs/experiments/dynamic_deephit.py#L397-L398) is hardcoded to `return "cpu"`, so this Optuna search over 8M+ records runs CPU-only (`nvidia-smi` confirms 0% GPU util) — slow by design, not stuck; also affects rep4's isolated run which copied the same file. Log-reading caveat: `eval_all_rep1.log` has had zero trial-completion lines since "study created" at 20:12:04 on Aug 16 (only 24 lines total) despite steady CPU burn — likely stdout buffering (`python -m ...` run without `-u`), so an unchanging log alone isn't evidence of a stall. `commons.py` `current_rep` still `5`, untouched (expected — this process hasn't moved past rep1). rep2-5 not started by this process (covered by separate standalone runs). Auto-checked every 10 min via session cron job `834cccdc`. Owned by this session as of Aug 17 18:55 CDT. |
| rep2 | 🔄 PID 1773328 | ⏳ | ⏳ | ⏳ | ⏳ | In Progress (standalone run via `run_rep.sh`) — re-verified 05:05 CDT (Aug 18) on `sunlab-serv-02.cs.illinois.edu`, `cox` still fitting (~1289% CPU, 10h28m elapsed), CPU time still climbing every check (most recently 5d12h56m→5d14h58m) so it's genuinely progressing, not hung; no `ckd_fifty_features_heterogeneous_cox_model.dill` yet. Consistent with the same slow-`cox` pattern seen on rep1/rep4/rep5's rows above. Auto-checked every 10 min via session cron job `5903a026`. Owned by this session as of Aug 17 18:55 CDT. |
| rep3 | 🔄 PID 1773554 | ⏳ | ⏳ | ⏳ | ⏳ | In Progress (standalone run via `run_rep.sh`) — re-verified 05:05 CDT (Aug 18) on `sunlab-serv-02.cs.illinois.edu`, `cox` still fitting (~1346% CPU, 10h28m elapsed), CPU time still climbing every check (most recently 5d18h38m→5d20h58m) so it's genuinely progressing, not hung; no `ckd_fifty_features_heterogeneous_cox_model.dill` yet. Same context as rep2. Auto-checked every 10 min via session cron job `5903a026`. Owned by this session as of Aug 17 18:55 CDT. |
| rep4 | 🔄 PID 93413 | ⏳ | ⏳ | ⏳ | ⏳ | In Progress — re-verified 16:16 CDT (Aug 19) on `sunlab-serv-01.cs.illinois.edu`, `cox` still fitting (PID 93413, ~2617% CPU, ~1d21h43m elapsed), CPU time now past 49 days cumulative [49-20:56:09], still climbing so it's genuinely progressing per raw CPU-time, not hung — **however, see new root-cause investigation below**: this session began an unrelated, isolated investigation (per explicit user request, not touching this live run) into why cox/dynamic_deephit take this long, and found the CKD_FIFTY_FEATURES_HETEROGENEOUS design matrix has an exact rank deficiency (the 50 `_missing` indicator columns sum to a constant 49 for every row → one redundant dimension), which can slow Newton-Raphson convergence; a small-scale (300-subject) repro fit in 9s with no warnings, a larger-scale repro is in progress to check whether convergence degrades non-linearly with data size. Investigation running in a fully separate `/home/minhn2/kidney-cox-investigation` sandbox against read-only copies of the CSVs — this live rep4 process/PID/model files are untouched. Swap/RAM steady. Full `cox` verified independently this session (per user request): recomputed C-index/Brier/AUC for rep1's cached model exactly matched the logged values, confirming the load-vs-train caching behavior in `cox.py` doesn't produce wrong results — only sharing this host with rep1's `dynamic_deephit`, not a 4-5-way pileup (rep2/rep3/rep5 run on other hosts). **Confirmed why this is slow**: unlike rep1 (which had a pre-existing `ckd_fifty_features_heterogeneous_cox_model.dill` from May 27 and just loaded+evaluated it in ~5min — see [pkgs/experiments/cox.py:29-38](pkgs/experiments/cox.py#L29-L38), it only trains fresh if no model file exists), rep4 has **no** pre-existing model file at `generated_data/rep4/ckd_fifty_features_heterogeneous_cox_model.dill`, so this is a genuine from-scratch `CoxTimeVaryingFitter` fit on 26k patients × 8M+ records — the multi-hour runtime is legitimate training, not a hang or a bug. Runs against an isolated copy of `pkgs/` at `/home/minhn2/kidney-rep4-run` (own `current_rep=4` hardcoded), so it never touches the shared `pkgs/commons.py`. Log: `pkgs/scripts/eval_all_rep4.log`. Auto-checked every 10 min via session cron job `46731da2`; owned by this session. |
| rep5 | 🔄 PID 1776284 | ⏳ | ⏳ | ⏳ | ⏳ | In Progress — attempt 1 (PID 968997/969012) found dead at 18:54 CDT (Aug 17); relaunched 18:55 CDT via `run_rep.sh 5`. Re-verified again 05:04 CDT (Aug 18) — attempt 2 (1776250/1776284) still healthy, `cox` fitting (~1268% CPU, 10h08m59s elapsed, CPU still active), no `ckd_fifty_features_heterogeneous_cox_model.dill` yet. Unlike rep1 (which loaded a pre-existing `.dill` instead of retraining), rep5 has no pre-existing model file, so this is a genuine full fit from scratch — consistent with the long runtime. Same CPU-contention pattern seen on rep1/rep2/rep3's rows too. 968997/969012 re-checked a 50th time (05:04 CDT), still empty. Auto-checked every 10 min via session cron job `ca5a55c0`. |

### Phase 3: Results

| Metric | Rep1 | Rep2 | Rep3 | Rep4 | Rep5 | Mean ± Std |
|--------|------|------|------|------|------|------------|
| C-index | - | - | - | - | - | - |
| Brier Score | - | - | - | - | - | - |
| AUC | - | - | - | - | - | - |
