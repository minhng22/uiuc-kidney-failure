
Analyses reuse saved models and automatically train missing selected models:

```bash
# all three analyses, all scenarios, reps 1-5
python -m pkgs.scripts.run_experiments analyze --reps all --parallel-reps

# one analysis, selected reps
python -m pkgs.scripts.run_experiments analyze --reps 2 3 --analyses clinical_validity
```

The default includes:

- `clinical_validity`: calibration, patient bootstrap intervals and paired differences
  (Gap 3), corrected patient-level IPCW/Brier/AUC and treat-all/eGFR decision curves
  (Gaps 5a–5d), including the recent numerical-ranking and common-cohort fixes.
- `feature_importance`: existing SHAP reports and charts.
- `subgroup`: age/sex/race performance with bootstrap intervals (Gap 12).

Repetitions run in parallel; the three tasks within each repetition run sequentially,
with separate logs. Model evaluations train missing models from the existing
train/test exports first.
Missing KFRE score caches are generated where applicable. Training appears in the
analysis task's log; a training failure makes the command exit nonzero.
`CKD_N_BOOTSTRAP` defaults to 1000.

Analysis uses existing exports and reuses existing models. Applying backward-only uACR matching to
the reported model results requires re-extraction and retraining first.

Then aggregate discrimination metrics (mean ± SD across whichever reps have
a `<scenario>_clinical_validity_report.txt`):

```bash
PYTHONPATH=. python -m pkgs.scripts.aggregate_rep_metrics four_features eight_features twenty_features_heterogeneous
```

Production data, model artifacts, reports and logs use `generated_data/rep_1/`
through `generated_data/rep_5/`; rep99 and rep100 retain their existing names.
Reports and charts are
overwritten on rerun, including by subset runs — use the full selection for
final comparisons.

`python -m pkgs.scripts.run_experiments --help` lists every option.

To run this script in the background, do:

```bash
nohup python -u -m pkgs.scripts.run_experiments analyze --reps 1 --parallel-reps > generated_data/run_experiments_rep_1.log 2>&1 < /dev/null &
```
