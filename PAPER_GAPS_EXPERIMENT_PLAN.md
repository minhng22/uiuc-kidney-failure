# Paper Gaps — Status + Fix Plan

**Last Updated:** 2026-09-18 (sunlab-serv-01) — plan only, nothing launched.

Gaps 3–13 from [paper/to submit 2026/README.md](paper/to%20submit%202026/README.md),
re-verified against the code and data.

**Context:** `generated_data/rep1`–`rep5` were rebuilt today (data only, three
splits per scenario, its own random seed per repetition). All model artifacts
behind the paper's results are gone, so a **full retrain is forced**. Fix the
before-retrain items first, the after-retrain items second. Cox's 0.308
concordance index is a covariate leak already fixed in `cox.py`; the retrain
clears it.

---

**Gap 3 — no uncertainty quantification**
- relevant (narrowed: the "one fixed patient partition" half is closed by the rebuilt repetitions)
- Fix (after retrain): bootstrap test-set patients about 1000 times, report percentile 95 percent confidence intervals for the concordance index, the integrated Brier score, and the time-dependent area under the curve; take paired differences against the Kidney Failure Risk Equation and against the best model on the same resamples. Predictions are computed once and re-indexed, so this costs minutes. Tables change from mean and standard deviation to mean with a confidence interval.

**Gap 4 — Logistic Hazard tunes on the test set**
- relevant — `val_data=(x_test, y_test)`, [logistic_hazard.py:82](pkgs/experiments/logistic_hazard.py#L82)
- Fix (before retrain): carve a stratified, patient-level validation slice out of the training split and use it for trial selection and checkpointing. The new three-way repetitions make this clean.

**Gap 5 — evaluation-unit mismatches**
- relevant — the censoring reference for inverse probability of censoring weighting is built from raw training rows ([clinical_validity_analysis.py:419](pkgs/data_analysis/clinical_validity_analysis.py#L419), `:474`) while predictions are one per patient; the integrated Brier score uses a shared scalar-risk transformation and never the native survival distribution (`native_prob_fn`); decision curve comparators are row-level (`:722`, `:729`) while the model curves are patient-level
- Fix (before retrain): build the censoring reference from the one-row-per-patient training frame and assert the counts match; thread `native_prob_fn` into the Brier path and label per model which was used; compute the decision curve comparators from the same patient-level frame. The last one restores the withdrawn net-benefit claim.

**Gap 6 — clarify population and outcome differences in the KFRE comparison**
- Already partly covered: the [PLOS manuscript's Limitations](paper/to%20submit%202026/paper%20content/plos_digital_health.tex#L371) acknowledge that the selected hospital cohort differs from KFRE's validation populations and that results should not generalize to routine clinical screening. The issue is how those differences affect interpretation of the benchmark, not an absence of any discussion or evidence that the comparison is inherently invalid.
- Correct the earlier characterization: the [original KFRE study](https://pubmed.ncbi.nlm.nih.gov/21482743/) used nephrology-referred CKD stage 3–5 cohorts, with kidney-failure event proportions of 11% and 24%; [subsequent multinational validation](https://jamanetwork.com/journals/jama/fullarticle/2481005) extended beyond that setting. Remove the blanket claim of incidence "one to two orders of magnitude lower." The benchmark's 83–92% ESRD-positive proportion is not automatically comparable to a fixed-horizon risk or an incidence rate; outcome definitions and follow-up differ.
- Next step (interpretation and reporting): verify and describe differences in baseline eligibility, outcome definition, and prediction horizon. In particular, the original KFRE endpoint was need for dialysis or preemptive transplantation, whereas this extraction labels ESRD using diagnosis codes. Explain that learned models are fitted to the selected MIMIC cohort while the published KFRE equation is applied without refitting. These differences may affect calibration and discrimination, but do not establish the cause or magnitude of any performance gap. Limit superiority claims to the evaluated cohort and protocol; coordinate prediction-time checks with Gap 8.

**Gap 8 — verify alignment of prediction time, available inputs, and outcome horizon**
- Existing design and rationale: [time_series_store.py](pkgs/data_analysis/time_series_store.py#L348) assembles each four-/eight-feature row around a creatinine measurement. In the eight-feature scenario, calcium, phosphate, bicarbonate, and serum albumin are each matched to the nearest measurement in the same admission, up to 24 hours before or after that timestamp. The [merge-design rationale](EXPERIMENT_PLAN_DETAILS.md#L99) uses this window to approximate a clinical snapshot from asynchronously measured labs, citing APACHE II as motivation. The urine albumin-to-creatinine ratio (uACR) is matched across the patient's whole history without a maximum time separation: the [pilot](generated_data/rep100/stage1c0_pilot_extraction_report.txt) found that the original same-admission rule excluded 96.5% of patients, disproportionately excluding ESRD-negative patients. These are documented, accepted extraction tradeoffs, not unexplained implementation mistakes.
- Remaining concern: the matching rules permit inputs recorded after the creatinine timestamp. Audit each model's actual prediction time, input cutoff, and outcome definition before interpreting its score as prospective risk. An approximate retrospective snapshot does not by itself establish what information would have been available at a clinical decision time. APACHE II's first-24-hour observation period is an analogy, not direct validation of a symmetric matching window around every creatinine measurement. Likewise, retaining a common time origin is legitimate for time-varying survival models with start/stop intervals; durations do not universally need resetting at every row.
- Next step: document the prediction target and check temporal alignment for each model, including the timing of matched labs relative to both the prediction time and the outcome, and how follow-up supports the reported risk horizon. Quantify later matches and time separations to assess the limitation. For a retrospective benchmark, describe the reconstruction and limit claims accordingly; disclosure alone does not correct any leakage found. If prospective prediction is the intended claim, define an information cutoff and align eligible patients, inputs, and subsequent outcomes with it, then assess the extraction and retraining changes required. Do not prescribe backward matching, duration resets, or a full re-extraction before that audit.

**Gap 12 — demographic descriptions exist; model performance by demographic subgroup remains unevaluated**
- Already covered: [demographics.py](pkgs/data_analysis/demographics.py#L53) implements age, gender, and race distributions for ESRD, non-ESRD, and CKD cohorts. [cohort_flow_analysis.py](pkgs/data_analysis/cohort_flow_analysis.py#L96) summarizes age and sex for source and extracted cohorts; saved [four-feature](generated_data/rep100/four_features_cohort_flow_report.txt) and [twenty-feature](generated_data/rep100/twenty_features_heterogeneous_cohort_flow_report.txt) reports confirm these summaries were produced. The blanket claim that demographic analysis was not performed was too broad.
- Remaining gap: no demographic-stratified discrimination or calibration was found in the clinical-validity analyzer or inspected reports. Cohort composition, including descriptions separated by ESRD status, does not measure model performance within age, sex, or race groups. The experiment runner currently offers only `clinical_validity` and `feature_importance`; `subgroup` would be a new analysis option.
- Extension (after retrain): reuse the existing demographic utilities, joining one metadata record per held-out patient, and compute per-group patient/event counts, event proportions, concordance, and survival-probability metrics where follow-up and sample size support them, with uncertainty intervals. Define age at the prediction landmark explicitly rather than treating `anchor_age` as landmark age. Audit race handling before reuse: [get_admission_df](pkgs/data_analysis/store.py#L33) drops unknown/declined/unavailable categories, and [ethnicity_and_race_statistics](pkgs/data_analysis/demographics.py#L78) keeps the first remaining admission row per patient. Preserve missing/unknown coverage and document the patient-level race selection rule. Report this as subgroup performance analysis; it alone does not establish fairness.

---

## TODO For PLOS submission


**TODO 1 — reporting improvement for the survival-prediction benchmark**
- Relevant — TRIPOD+AI applies because the hazard and survival models are evaluated for individual prognostic prediction, rather than solely for associations or hazard ratios. Its scope includes both regression and machine learning ([official scope](https://www.tripod-statement.org/scope/)).
- Recommendation: verify reporting completeness: STROBE is already cited for cohort flow. Audit the manuscript against applicable STROBE and TRIPOD+AI items, identify actual omissions, and supply a checklist as appropriate. Treat this as manuscript preparation, not an established methodological defect.

**TODO 2 — complete ethics and data-access reporting before submission**
- Relevant as a reporting fix: the lead [PLOS manuscript](paper/to%20submit%202026/paper%20content/plos_digital_health.tex#L411) has only a commented-out ethics statement; the [Springer version](paper/to%20submit%202026/paper%20content/sn-article.tex#L73) has generic wording. [PLOS submission guidelines](https://journals.plos.org/digitalhealth/s/submission-guidelines#loc-human-subjects-research) call for an ethics statement in Methods identifying approval or explaining why it was not needed, and addressing informed consent.
- Fix (text, before submission): describe the secondary use of de-identified data, cite the source dataset's documented ethics approval and consent waiver, and state the verified approval/exemption basis applicable to this analysis. Distinguish the source dataset's oversight from any determination covering this study; do not assume approval by particular institutions or invent an exemption. This is not, by itself, a finding that new approval or experiments are required.
- Data access and reproducibility: identify the MIMIC-IV version actually used and its DOI, and explain access through PhysioNet's credentialing, required training, and data-use agreement. These are dataset-access conditions; no blanket PLOS requirement to recite the exact CITI course title was found. Include author-specific credential/training claims only when verified.

**Separate bibliography cleanup (not an ethics gap)**
- `ishwaran2008random` lacks a journal field. Check `lee2018deephit` against the selected bibliography style before resolving the reported volume/number warning. `hu2022locf_bias` already has an author and is an arXiv `@misc` entry; the earlier missing-author/publisher claim should not be carried forward without a current check.

