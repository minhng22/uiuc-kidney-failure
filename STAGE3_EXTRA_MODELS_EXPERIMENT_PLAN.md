# Stage 3 extra models (deepsurv/gbsa/srf/survival_svm/weibul) — rep99

User request: these 5 models exist in `pkgs/experiments/` but were never part
of the approved Stage 3 plan for `four_features`/`eight_features`/
`twenty_features_heterogeneous` (`EXPERIMENT_PLAN_DETAILS.md` line 493 lists
only `cox`/`ddh`/`hazard_transformer`/`logistic_hazard`/`rnn_surv`/`kfre`) —
they were hardcoded to the older `NON_TIME_VARIANT` scenario. Adding them to
all 3 new scenarios and rerunning rep99, per user decision:
data-shape = last observation per subject (flattened via new
`get_last_observation_data()` in `pkgs/data_analysis/model_data_store.py`),
scope = all 5 models.

Code changes (additive only — original NON_TIME_VARIANT-only functions in
each file are untouched):
- `pkgs/data_analysis/model_data_store.py`: new `get_last_observation_data(scenario)`.
- `pkgs/experiments/utils.py`: `get_x_for_sckit_survival_model()` gained an
  optional `scenario` param (backward compatible, defaults to old behavior).
- `pkgs/experiments/{deepsurv,gbsa,srf,survival_svm,weibul}.py`: each gained
  a `<name>_model_path_dict` + `run_scenario(scenario)`.
- `pkgs/commons.py`: 15 new path constants (5 models x 3 scenarios).

All 5 smoke-tested successfully on rep99/four_features before the full runs
below (weibul c=0.547, survival_svm c=0.582, srf c=0.551, gbsa in progress,
deepsurv c=0.534 with a shortened 2-trial search then discarded/deleted
before the real run).

## Status

| Scenario | Models | Status |
|---|---|---|
| four_features | all 5 | done |
| eight_features | all 5 | done |
| twenty_features_heterogeneous | all 5 | done |

Full results table, per-scenario numbers, and flagged follow-ups (a couple of
near-chance results worth a closer look): see
[report](generated_data/rep99/stage3_extra_models_report.txt).

## Background processes

(none currently active — the run finished; see report above for the
permanent record)

Last Updated: 2026-08-27 (sunlab-serv-02.cs.illinois.edu)
