# Running model evaluation

All commands are run from the repository root with the project conda
environment active (built from [environment.yml](../../environment.yml)).
`python -m` puts the repo root on `sys.path`, so `PYTHONPATH` only needs
setting when invoking from elsewhere.

Which rep is read/written is bound at import time by `CKD_REP`
(see [commons.py:155](../commons.py#L155)) — **it defaults to `rep5`, not
`rep1`**, so never run an experiment module without setting it explicitly.
[run_experiments.py](../scripts/run_experiments.py) sets `CKD_REP` itself per
worker, which is why it is the preferred entry point below.

# To test a model on a rep, run

```bash
python -m pkgs.scripts.run_experiments train --reps 1 --models cox
```

Scope it further with `--scenarios` (any of `four_features`,
`eight_features`, `twenty_features_heterogeneous`; all three by default):

```bash
python -m pkgs.scripts.run_experiments train --reps 1 --models cox --scenarios four_features eight_features
```

`--models` accepts: `cox`, `dynamic_deephit`, `hazard_transformer`,
`logistic_hazard`, `rnnsurv`, `kfre`, `deepsurv`, `gbsa`, `srf`,
`survival_svm`, `weibul`. KFRE is skipped on
`twenty_features_heterogeneous` (no published equation for it).

This calls each model's run function directly
([run_experiments.py:94-104](../scripts/run_experiments.py#L94-L104)) and
fails a scenario whose `<scenario>_{train,test}_data.csv` is missing under
`generated_data/rep<N>/`, rather than silently falling through to a raw
MIMIC extraction. Logs land in
`generated_data/rep<N>/rep<N>_train_<timestamp>.log`; the runner stays in
the foreground. Add `--dry-run` to print the worker commands first.

**Do not** reach for `python -m pkgs.experiments.<model>` to test one model
on one rep. Each module's `__main__` block hardcodes all three scenarios,
and several run extra legacy scenarios on top: [deepsurv.py](deepsurv.py)
calls `run()` first, [weibul.py](weibul.py) calls `run_ti()`, and
[survival_svm.py](survival_svm.py) calls `run_all()`, which also runs the
non-time-variant model. Read the `__main__` block before invoking a module
directly (see the entry-point rule in [CLAUDE.md](../../CLAUDE.md)).

# To test all models on a rep, run

Foreground, sequential, all 11 models × 3 scenarios:

```bash
python -m pkgs.scripts.run_experiments train --reps 1
```

Or, to run the models **in parallel in the background** (one subprocess and
one log per model), use [run_rep.sh](../scripts/run_rep.sh):

```bash
pkgs/scripts/run_rep.sh 1
```

It selects the rep via `CKD_REP` (no shared state mutated), so several reps
can be launched concurrently. It writes:

| file | contents |
| --- | --- |
| `pkgs/scripts/eval_all_rep<N>.log` | launch/completion line per model + final summary |
| `pkgs/scripts/eval_rep<N>_<model>.log` | that model's own stdout/stderr |
| `pkgs/scripts/run_rep<N>_master.log` | wrapper output |
| `pkgs/scripts/run_rep<N>.pid` | PID of the detached process group |

Useful env vars: `EXCLUDE_EXPERIMENTS="dynamic_deephit"` to skip a model
already running elsewhere for that rep, and `RUN_TAG=resume` to suffix this
invocation's log/pid files so it doesn't clobber a live run's.

Monitor with `tail -f pkgs/scripts/eval_all_rep<N>.log`; stop with
`kill -TERM -$(cat pkgs/scripts/run_rep<N>.pid)`.

Because this is a background process, record its PID, host, launch command,
log path and start time in the relevant `*_EXPERIMENT_PLAN.md` — see
[CLAUDE.md](../../CLAUDE.md).

# To test all models on all reps, run

```bash
python -m pkgs.scripts.run_experiments train --reps all --parallel-reps
```

`--reps all` means the production reps 1-5; rep99 (the mini-experiment rep)
must be selected explicitly (`--reps 99`). `--parallel-reps` runs the reps
concurrently — tasks within a rep stay sequential — and each rep gets its
own log under `generated_data/rep<N>/`. Drop the flag to run reps one after
another. A failed worker doesn't stop the rest; the runner exits non-zero if
any failed.

To get the background/parallel-per-model behaviour for every rep, launch
[run_rep.sh](../scripts/run_rep.sh) once per rep instead:

```bash
for rep in 1 2 3 4 5; do pkgs/scripts/run_rep.sh "$rep"; done
```

[run_all_reps.sh](../scripts/run_all_reps.sh) also exists but is legacy:
it covers only 5 of the 11 models, runs everything strictly sequentially,
and mutates `current_rep` in [commons.py](../commons.py) in place — so it
cannot be run alongside any other rep's run. Prefer one of the two commands
above.

# After a run

Analyses consume the trained model artifacts written above:

```bash
# both analyses, all scenarios, reps 1-5
python -m pkgs.scripts.run_experiments analyze --reps all --parallel-reps

# one analysis, selected reps
python -m pkgs.scripts.run_experiments analyze --reps 2 3 --analyses clinical_validity
python -m pkgs.scripts.run_experiments analyze --reps 2 3 --analyses feature_importance
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
