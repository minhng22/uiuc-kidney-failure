
Analyses reuse saved models and automatically train missing selected models:

```bash
# all three analyses, all scenarios, reps 1-5
python -m pkgs.scripts.run_experiments analyze --reps all --parallel-reps
```

See params in pkgs/scripts/run_experiments.py

To run this script in the background, do:

```bash
nohup python -u -m pkgs.scripts.run_experiments analyze --reps 2 3 4 5 --scenarios four_features eight_features > generated_data/run_experiments_rep_2_3_4_5.log 2>&1 < /dev/null &
```
