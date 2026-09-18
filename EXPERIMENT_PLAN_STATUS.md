# Relevant gaps — implementation progress

Gaps as listed in [PAPER_GAPS_EXPERIMENT_PLAN.md](PAPER_GAPS_EXPERIMENT_PLAN.md).
Findings and numbers live in the per-stage report files linked below, not here.

**Last Updated:** 2026-09-18 (sunlab-serv-01)

Gap 3 — patient bootstrap uncertainty
- state: done (code + rep99 verification); production numbers blocked on rep1-5 retraining
- status details: [pkgs/data_analysis/bootstrap_ci.py](pkgs/data_analysis/bootstrap_ci.py) resamples test
  patients 1000x on one shared index matrix; predictions are computed once and re-indexed. C-index / IBS /
  mean time-dependent AUC get 95% percentile intervals, plus paired differences against KFRE and against
  the best model on the same resamples. Wired into the clinical-validity report; resample count via
  `CKD_N_BOOTSTRAP`. Verified on rep99, all 3 scenarios, 11 models — see
  [report](generated_data/rep99/stage_gaps_evaluation_rework_report.txt). Measured cost 8 ms/resample/model.
  Blocked on: reps 1-5 hold data CSVs but no trained artifacts.

Gap 5a — patient-level censoring reference
- state: done (code + rep99 verification)
- status details: [pkgs/data_analysis/patient_outcomes.py](pkgs/data_analysis/patient_outcomes.py) builds and
  verifies one terminal outcome per patient; the IPCW reference is now built from it, not from raw lab-event
  rows. Per-model alignment is checked before use and all 11 models are scored against one canonical outcome
  set. Side effect found and fixed: the time-dependent AUC had been `None` for every model in every scenario
  because only the Brier path applied administrative censoring — all 11 now report an AUC. Verified on rep99 —
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
  model curves; the superseded row-level curve is still printed for contrast. This is the largest single
  correction in the rework (rep99 four_features, 730 d, pt=0.05: 0.032 row-level to 0.492 patient-level), so
  any clinical-utility claim resting on a model beating treat-all must be re-derived — see
  [report](generated_data/rep99/stage_gaps_evaluation_rework_report.txt).

Gap 5d — eGFR comparator
- state: done (code + rep99 verification)
- status details: one referral decision per patient, from their most recent genuinely measured eGFR at or
  before their prediction landmark, carrying their terminal outcome; coverage and exclusions reported
  explicitly (rep99: full coverage in all 3 scenarios; twenty_features selects 20 decisions from 502
  measured-eGFR rows out of 9115). The rule is also now evaluated across the SAME `DCA_THRESHOLDS` as the
  model curves and plotted as a line on the same axis, replacing the single point at its own
  referral-fraction "implied threshold" — the second half of the gap ("compare strategies at the same
  decision thresholds"). See
  [report](generated_data/rep99/stage_gaps_evaluation_rework_report.txt).

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
  future extractions record this too. No matching rule, duration reset or re-extraction was changed. Run on
  rep99 and on the production rep1 cohort; the rep1 numbers are the ones now quoted in the manuscripts'
  prediction-time limitation. See
  [rep1 report](generated_data/rep1/stage_gap8_prediction_time_audit_report.txt) and
  [rep99 report](generated_data/rep99/stage_gap8_prediction_time_audit_report.txt).
- open follow-up: the audit describes the reconstruction; deciding whether the manuscript claims prospective
  risk (which would require defining and enforcing an information cutoff, then re-extracting) is a framing
  decision the audit deliberately does not make.

Gap 12 — subgroup performance
- state: done (code + rep99 verification)
- status details: new `--analyses subgroup` option
  ([pkgs/data_analysis/subgroup_analysis.py](pkgs/data_analysis/subgroup_analysis.py),
  [pkgs/data_analysis/patient_metadata.py](pkgs/data_analysis/patient_metadata.py)) reports per-group patient
  and event counts, C-index, IBS and mean AUC with bootstrap intervals, over age / sex / race. Race handling
  audited before reuse: unknown/declined is retained as its own group rather than dropped, and the
  patient-level value is modal-across-admissions rather than first-row. Age is reported as `anchor_age` under
  its own name, not relabelled as landmark age. Reported as subgroup performance, not fairness. See
  [report](generated_data/rep99/stage_gap12_subgroup_implementation_report.txt); per-scenario numbers in
  `generated_data/rep99/<scenario>_subgroup_performance_report.txt`.
