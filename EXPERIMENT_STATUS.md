# Feature Set Experiment Plan (4/8/20 features)

Tracks execution of [EXPERIMENT_PLAN_DETAILS.md](EXPERIMENT_PLAN_DETAILS.md) (Stage 0 plan,
approved). Do not restart another session's row without confirming its host is actually dead.

## Status

| Stage | Task | Status | Notes |
|---|---|---|---|
| 1a | Task A: determine 20 lab features | done | [report](generated_data/rep1/twenty_features_lab_analysis_report.txt) |
| 1a | Task B: locate Tangri et al. 8-variable KFRE coefficients | done | [report](generated_data/rep1/kfre_8variable_coefficients_report.txt) |
| 1b | Code changes (types/commons/time_series_store/model_data_store/experiments/kfre) | done | [report](generated_data/rep1/stage1b_implementation_report.txt) |
| 1c-0 | Pilot extraction (rep1) + cohort-flow analysis — approval gate | done, approved | [report](generated_data/rep1/stage1c0_pilot_extraction_report.txt) |
| 1c | Full extraction (rep2-5, parallel) | done | [report](generated_data/rep1/stage1c_full_extraction_report.txt) |
| 2 | Mini-experiment (rep99) | **done** — all rep99 models deleted and retrained from scratch (picks up DDH/HazardTransformer/RNN-Surv/DeepSurv fixes + the 5 newly-added models); 12/17 official-driver combos clean, 5 hit the same known four_features AUC-censoring-sparsity edge case (models still trained/saved fine — see Background processes) | [report](generated_data/rep99/mini_experiment_status_report.txt) |
| 2.1 | Feature-importance analysis + additional analyses (calibration + decision-curve, rep99 sanity check) | **done** — rerun 2026-08-27 against the retrained rep99 models (Stage 2 above); clean, no errors | SHAP reports: [four_features](generated_data/rep99/four_features_shap_analysis_report.txt), [eight_features](generated_data/rep99/eight_features_shap_analysis_report.txt), [twenty_features_heterogeneous](generated_data/rep99/twenty_features_heterogeneous_shap_analysis_report.txt); clinical-validity reports: [four_features](generated_data/rep99/four_features_clinical_validity_report.txt), [eight_features](generated_data/rep99/eight_features_clinical_validity_report.txt), [twenty_features_heterogeneous](generated_data/rep99/twenty_features_heterogeneous_clinical_validity_report.txt); charts: `<scenario>_calibration_plot.png` / `<scenario>_decision_curve_plot.png` per scenario, plus cross-model `c_index_comparison.png` / `brier_comparison.png` / `auc_comparison.png`, all in `generated_data/rep99/` |
| 2 | PMF-fix rerun (rep99) | done | [report](generated_data/rep99/stage2_pmf_fix_rerun_report.txt) |
| 2.1 | PMF-fix rerun analyses (rep99) | done | [report](generated_data/rep99/stage2_pmf_fix_rerun_report.txt) |
| 2.2 | Debug self-check on Stage 2.1's rep99 analysis | **Finding #1 fixed, verified, rerun (2026-08-28)** — eGFR-referral-rule net benefit bug for twenty_features_heterogeneous; Stage 2.1's clinical-validity report regenerated for rep99 (model retraining skipped — fix doesn't touch models). **Finding #2 still open** (generic risk-score→probability transform saturates for 6/11 models' calibration/Brier/DCA numbers in every scenario) — awaiting user direction on 1 of 3 proposed options. 2 open questions (#3/#4) not yet root-caused | [report](generated_data/rep99/stage2_2_debug_report.txt) |
| 3.0 | rep1 four_features/eight_features run (all 11 models) — analysis report, then approval gate before 3.1 and separately before `twenty_features_heterogeneous` (per Stage 3.0's scenario-ordering rule) | **run + analysis report both done** 2026-08-28, no errors — but `dynamic_deephit`'s `eight_features` model was clobbered mid-run by a second, unidentified writer (unresolved caveat carries into the analysis too — see [report](generated_data/rep1/stage3_0_rep1_run_report.txt)); **awaiting user review + approval gate** before Stage 3.1 or `twenty_features_heterogeneous` | see Background processes below |
| 3.1 | rep2-4 full runs (blocked on 3.0's approval gate) | **not done** — rep2/rep3/rep4 were started (2 sessions) before Stage 3 was split into 3.0/3.1; all stopped/killed by user request before finishing; needs relaunch once 3.0 is approved | see Background processes below |

## Background processes

(Hazard Transformer horizon-fix rerun on rep99, PID 4190139 on
sunlab-serv-02, confirmed finished 2026-08-27 — log
[stage2_ht_horizonfix_rep99.log](pkgs/scripts/logs/stage2_ht_horizonfix_rep99.log)
ends in "DONE" for all 3 scenarios, and the Stage 2.1 rep99 reports it
queued a rerun for are timestamped after its completion. Row dropped now
that it's finished; see [Stage 2.2 debug report](generated_data/rep99/stage2_2_debug_report.txt)
for the verification detail.)

(Stage 3.0's rep1 four_features/eight_features run, PID 2697933 on
sunlab-serv-02, finished 2026-08-28 05:56 CDT — 11/11 done, no errors. Row
dropped now that it's finished; full story, including a second clobbering
incident mid-run, in [report](generated_data/rep1/stage3_0_rep1_run_report.txt).)

Launch/reuse-plan details: [report](generated_data/rep1/stage3_0_rep1_run_report.txt).
`get_device()` GPU-selection change + parallel experiment launching + this resume
run: [report](generated_data/rep1/stage3_0_parallel_resume_report.txt).

Ad-hoc `ddh`/`four_features`+`eight_features` eval (user request, PID 2042631, separate from
the main rep1 run above) — done, see [report](generated_data/rep1/four_eight_features_ddh_eval_report.txt).

TEMPORARY/interim Stage 2.1-style analysis for rep1 (user request, everything trained so
far — not the final report) — done, see [check log](generated_data/rep1/stage3_0_background_process_log.txt)
for full checklist; reports/charts alongside the rep99 ones in `generated_data/rep1/`.

Last Updated: 2026-08-27 (sunlab-serv-01.cs.illinois.edu) — relaunched Stage
3.0 rep1 run as PID 2458050 with the widened 11-model scope (ddh/
hazard_transformer/rnnsurv/deepsurv/gbsa/srf/survival_svm/weibul; cox/kfre/
logistic_hazard reused from the earlier completed run). Verified clean start
for all 8: no immediate errors, all subprocess PIDs alive ~30s in.

10-min health check (2026-08-27): `survival_svm` completed successfully (all
4 scenarios). `weibul` crashed on `twenty_features_heterogeneous` —
`lifelines.exceptions.ConvergenceError` (near-complete separation from a
low-variance `*_missing` indicator column at rep1's full 26080-subject
scale; `four_features`/`eight_features` weibul both converged fine and
saved). Diagnosed, fixed (`penalizer=0.1` in `WeibullAFTFitter()`, scoped to
`run_scenario()` only — the original NON_TIME_VARIANT `run_ti()` path is
untouched), and verified read-only against rep1's actual training data
(reproduced the failure, confirmed the fix converges) before applying —
skipped rep99 re-verification because rep99's smaller sample never
triggered this failure to begin with (its `twenty_features_heterogeneous`
weibul model already exists, trained successfully pre-fix) and because
another session's rep99 job for these same models is still marked
in-progress in this doc's Stage 2 rerun row, so rep99 state wasn't touched.
Relaunched weibul alone as PID 2462750 (reuses the two already-saved
scenarios, only retrains twenty_features_heterogeneous). The other 6 models
(ddh/hazard_transformer/rnnsurv/deepsurv/gbsa/srf) still running under PID
2458050, no errors so far.

2nd 10-min health check (2026-08-27): weibul retry (PID 2462750) finished —
`✓ weibul completed successfully`, twenty_features_heterogeneous C-index
0.525 (fix confirmed working end-to-end, not just in the read-only test).
Dropped that row; weibul is now fully done for rep1 (all 4 scenarios). The
other 6 models under PID 2458050 all still alive, no tracebacks/errors in
any per-experiment log — hazard_transformer and deepsurv are multiple optuna
trials into their 2nd/3rd scenario respectively; dynamic_deephit/rnnsurv/gbsa
haven't logged a completed trial yet (still mid first trial, ~26min in) but
show normal CPU activity, no sign of a hang.

3rd 10-min health check (2026-08-27): `deepsurv` finished — `✓ deepsurv
completed successfully`, all 3 new scenarios ran (confirmed via scenario
markers in its log), C-index 0.529/twenty_features_heterogeneous. Dropped
that row. Remaining 5 (dynamic_deephit/hazard_transformer/rnnsurv/gbsa/srf)
all still alive under PID 2458050, no errors, all showing forward progress
since the prior check (srf on its 3rd grid-search round, hazard_transformer
onto a 2nd optuna study/scenario, dynamic_deephit finished its first trial).
6 of 11 models done for rep1 so far.
Note (2026-08-27, ~13 checks later): `hazard_transformer`'s current optuna
trial (2nd study, `twenty_features_heterogeneous` scenario, started
02:37:49) has now run for **~3.5+ hours** without completing — flagging
this explicitly as an unusually long single trial. Not treating it as
stalled: CPU-time has grown steadily and consistently (~357-390 CPU-min per
~11-12 wall-clock min, ~33 threads worth) across every ~10-min check since
it started, with no drop-off. Its log shows no `print()`-based progress
lines at all (confirmed buffering artifact — even fast completed trials in
this same log never showed them, only optuna's own `Trial N finished`
lines), so silence here carries no information either way; CPU-time is the
only reliable signal and it says this is real, ongoing computation — likely
an expensive hyperparameter draw (many layers/heads) on the 6.5M-row
`twenty_features_heterogeneous` scenario. Will keep monitoring; not killing
a process that's still doing confirmed real work just because it's slow.
The other 4 models (dynamic_deephit/rnnsurv/gbsa/srf) remain healthy, no
errors, dynamic_deephit/rnnsurv both approaching the end of their first
(four_features) 10-trial optuna search.

Last Updated (PMF rerun): 2026-08-27 18:00 CDT
(sunlab-serv-02.cs.illinois.edu) — rep99 Stage 2 retraining and both scoped
Stage 2.1 analyses completed; see
[report](generated_data/rep99/stage2_pmf_fix_rerun_report.txt).

Last Updated (Stage 3.0 restart): 2026-08-28 01:16 CDT
(sunlab-serv-02.cs.illinois.edu) — everything above this line about PID
2458050 is superseded: an untracked duplicate run collided with it and
deleted rep1's four_features/eight_features model files (full incident in
[report](generated_data/rep1/stage3_0_rep1_run_report.txt)); both process
trees killed, rep1 restarted from scratch as PID 2697933 via the new scoped
driver, per-scenario logs confirmed clean so far — see the Background
processes row above.

Last Updated (Stage 3.0 four/eight complete): 2026-08-28 05:56 CDT
(sunlab-serv-02.cs.illinois.edu) — PID 2697933 finished, 11/11 models done,
no errors. A second clobbering incident hit `dynamic_deephit`'s
`eight_features` model mid-run (unidentified second writer, root cause not
found) — see [report](generated_data/rep1/stage3_0_rep1_run_report.txt) for
full detail and the caveat on using that specific model file. Analysis
report + Stage 3.0→3.1 approval gate still pending, not started.

Last Updated (Stage 2.2 debug): 2026-08-28 11:44 CDT
(sunlab-serv-01.cs.illinois.edu) — ran Stage 2.2's self-check against Stage
2.1's rep99 reports; 2 confirmed issues + 2 open questions, see
[report](generated_data/rep99/stage2_2_debug_report.txt) — awaiting user
direction before applying either proposed fix or proceeding further.

Last Updated (Finding #1 fix): 2026-08-28 16:13 CDT
(sunlab-serv-01.cs.illinois.edu) — applied the eGFR-referral-rule fix
(user-approved), verified against rep99 data, reran Stage 2.1's
clinical-validity driver for rep99 (model retraining skipped — fix is
analysis-code-only), confirmed the regenerated report and unchanged
discrimination metrics. Detail in
[report](generated_data/rep99/stage2_2_debug_report.txt). Finding #2 still
open, awaiting direction.
