
Analyses consume the trained model artifacts written above:

```bash
# both analyses, all scenarios, reps 1-5
python -m pkgs.scripts.run_experiments analyze --reps all --parallel-reps

# one analysis, selected reps
python -m pkgs.scripts.run_experiments analyze --reps 2 3
```

Then aggregate discrimination metrics (mean ± SD across whichever reps have
a `<scenario>_clinical_validity_report.txt`):

```bash
PYTHONPATH=. python -m pkgs.scripts.aggregate_rep_metrics four_features eight_features twenty_features_heterogeneous
```

Reports and charts are written under `generated_data/rep<N>/` and are
overwritten on rerun, including by subset runs — use the full selection for
final comparisons.

`python -m pkgs.scripts.run_experiments --help` lists every option.

To run this script in the background, do:

