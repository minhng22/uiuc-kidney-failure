# Relevant gaps — implementation progress

The per-gap descriptions below track evaluator and audit work.
Findings and numbers live in the per-stage report files linked below, not here.

**Last Updated:** 2026-09-23 (sunlab-serv-01)

Default evaluation command:
`python -m pkgs.scripts.run_experiments analyze --reps all --parallel-reps`
now runs clinical validity (Gaps 3, 5a–5d), feature importance, subgroup performance
(Gap 12), outcome-definition audit (Gap 6), and prediction-time audit including raw
lab timing (Gap 8). All three scenarios run for reps 1–5. Raw labs are scanned once
per repetition and shared across scenarios; each task has its own log. Use
`--analyses` to select a subset. Analysis does not re-extract or retrain; existing
Production paths are `generated_data/rep_1/` through `rep_5/`. Four-/eight-feature
exports have now been rebuilt with backward-only uACR; model retraining is pending.

Gap 3 — patient bootstrap uncertainty
- state: done (code + rep99 verification); production numbers blocked on rep1-5 retraining
- status details: [pkgs/data_analysis/bootstrap_ci.py](pkgs/data_analysis/bootstrap_ci.py) resamples test
  patients 1000x on one shared index matrix; predictions are computed once and re-indexed. C-index / IBS /
  mean time-dependent AUC get 95% percentile intervals, plus paired differences against KFRE and against
  the best model on the same resamples. Wired into the clinical-validity report; resample count via
  `CKD_N_BOOTSTRAP`. Verified on rep99, all 3 scenarios (11/11/10 models) — see
  [report](generated_data/rep99/stage_gaps_evaluation_rework_report.txt). Measured cost 8 ms/resample/model.
  Blocked on: reps 1-5 hold data CSVs but no trained artifacts.
  September 23 review: bootstrap C-index now retains full follow-up, matching its point estimate;
  only IBS/AUC use IPCW-capped outcomes. The same correction applies to subgroup analysis.
  Numerically constant rankings are explicitly withheld from points, intervals and paired rank
  differences; probability metrics remain available. See the linked report.

Gap 5a — patient-level censoring reference
- state: done (code + rep99 verification)
- status details: [pkgs/data_analysis/patient_outcomes.py](pkgs/data_analysis/patient_outcomes.py) builds and
  verifies one terminal outcome per patient; the IPCW reference is now built from it, not from raw lab-event
  rows. Per-model alignment is checked before use and all 11 models are scored against one canonical outcome
  set. Administrative censoring fixed the September 14 four-feature AUC failure; the eight- and
  twenty-feature runs already returned AUCs. The cap now uses sksurv's reverse-KM estimator and
  stops before its first zero, including correct handling of tied event/censoring times. Verified on rep99 —
  see [report](generated_data/rep99/stage_gaps_evaluation_rework_report.txt).

Gap 5b — survival probabilities for Brier evaluation
- state: done (code + rep99 verification)
- status details: each fitted estimator inspected rather than assumed score-only. Native survival curves now
  exposed for Cox (its own `baseline_cumulative_hazard_`), GBSA and Survival RF (`predict_survival_function`)
  and Weibull AFT (closed-form); Survival SVM and DeepSurv genuinely have none and keep the labelled fitted
  conversion; KFRE reports point Brier at its published 2 y / 5 y horizons instead of an invented curve.
  Every Brier number carries its source, with the fitted-conversion value printed alongside for contrast.
  8 of 11 models now report a native IBS — see
  [report](generated_data/rep99/stage_gaps_evaluation_rework_report.txt).

Gap 5c — treat-all comparator
- state: done (code + rep99 verification)
- status details: treat-all now uses the same patient-level outcomes, horizon and included patients as the
  model curves. A common finite-prediction/eGFR mask is enforced across all plotted DCA strategies at
  each horizon, with coverage reported; wholly unavailable models are omitted. The superseded row-level
  curve is still printed for contrast. This is the largest single
  correction in the rework (rep99 four_features, 730 d, pt=0.05: 0.032 row-level to 0.492 patient-level), so
  any clinical-utility claim resting on a model beating treat-all must be re-derived — see
  [report](generated_data/rep99/stage_gaps_evaluation_rework_report.txt).

Gap 5d — eGFR comparator
- state: done (code + rep99 verification)
- status details: one referral decision per patient, from their most recent finite, genuinely measured eGFR at or
  before their prediction landmark, carrying their terminal outcome; coverage and exclusions reported
  explicitly (rep99: full coverage in all 3 scenarios; twenty_features selects 20 decisions from 502
  measured-eGFR rows out of 9115). The rule is also now evaluated across the SAME `DCA_THRESHOLDS` as the
  model curves and plotted as a line on the same axis, replacing the single point at its own
  referral-fraction "implied threshold" — the second half of the gap ("compare strategies at the same
  decision thresholds"). See
  [report](generated_data/rep99/stage_gaps_evaluation_rework_report.txt).

Evaluation review regression checks:
`python -m unittest pkgs.data_analysis.test_evaluation_gaps` covers full-follow-up C-index resampling,
reverse-KM support with tied times, unresolved float32 rankings, missing eGFR/predictions, and common
DCA cohort orchestration. These close evaluator/reporting issues; the nearly constant twenty-feature
Hazard Transformer checkpoint is still unsuitable for a discrimination claim.

Gap 6 — KFRE population and outcome comparison
- state: done (audit + manuscript text)
- status details: [pkgs/scripts/audit_outcome_definition.py](pkgs/scripts/audit_outcome_definition.py) verifies
  the label and eligibility rule against the data. Confirmed: `esrd_codes` is entirely acute/unspecified
  kidney-failure codes; no ESRD or dialysis/transplant code is in it, and the event time is an admission
  timestamp. Outcome-definition text added to all three manuscript variants plus a Limitations item; the
  "orders of magnitude lower incidence" characterization is retracted with the reason. Run on rep99 and on
  the production rep1 cohort, so the horizon-support numbers quoted come from production data. See
  [rep1 report](generated_data/rep1/stage_gap6_outcome_definition_report.txt) and
  [rep99 report](generated_data/rep99/stage_gap6_outcome_definition_report.txt).

Gap 8 — prediction-time alignment audit
- state: done (audit); manuscript framing follow-up open
- status details: [pkgs/scripts/audit_prediction_time.py](pkgs/scripts/audit_prediction_time.py) documents each
  model's prediction target from its own code, quantifies horizon support from the exported CSVs, and (with
  `--lab-timing`) measures matched-lab timing against the anchor creatinine by calling the extraction's own
  merge. `merge_nearest_within_admission` now carries the matched timestamp and logs a `MATCH_TIMING|` line so
  future extractions record this too. The original audit changed no matching rule or duration reset. Run on
  rep99 and on the production rep1 cohort; the rep1 numbers are the ones now quoted in the manuscripts'
  prediction-time limitation. See
  [rep1 report](generated_data/rep1/stage_gap8_prediction_time_audit_report.txt) and
  [rep99 report](generated_data/rep99/stage_gap8_prediction_time_audit_report.txt).
- 2026-09-23 uACR fix: four-/eight-feature extraction now selects the latest uACR at or before
  each anchor creatinine, across admissions with no maximum lookback. Rows with no eligible uACR
  are dropped. The timing audit uses the same backward rule. Four matching/audit tests and the
  14 evaluator regression tests pass. Production four-/eight-feature exports were rebuilt in
  `generated_data/rep_1/` through `rep_5/`, including train/test/external-validation splits.
  Twenty-feature exports reuse the unchanged rep100 pool. Retraining and evaluation remain pending;
  rep99 still contains the earlier data/results. Chemistry still uses nearest +/-24 h.
  Cohort imbalance flagged in each rep's `cohort_balance_warning.txt`: label-positive patients are
  75.22% (four features), 76.88% (eight), and 91.45% (twenty). These are current `has_esrd`
  labels, not verified ESRD incidence. Four/eight proportions fell from 83.52%/85.08%.
- open follow-up: the audit describes the reconstruction; deciding whether the manuscript claims prospective
  risk (which would require defining and enforcing an information cutoff, then re-extracting) is a framing
  decision the audit deliberately does not make.

Gap 12 — subgroup performance
- state: done (code + rep99 verification)
- status details: `subgroup` now runs by default; `--analyses subgroup` runs it alone
  ([pkgs/data_analysis/subgroup_analysis.py](pkgs/data_analysis/subgroup_analysis.py),
  [pkgs/data_analysis/patient_metadata.py](pkgs/data_analysis/patient_metadata.py)) reports per-group patient
  and event counts, C-index, IBS and mean AUC with bootstrap intervals, over age / sex / race. Race handling
  audited before reuse: unknown/declined is retained as its own group rather than dropped, and the
  patient-level value is modal-across-admissions rather than first-row. Age is reported as `anchor_age` under
  its own name, not relabelled as landmark age. Reported as subgroup performance, not fairness. See
  [report](generated_data/rep99/stage_gap12_subgroup_implementation_report.txt); per-scenario numbers in
  `generated_data/rep99/<scenario>_subgroup_performance_report.txt`.
