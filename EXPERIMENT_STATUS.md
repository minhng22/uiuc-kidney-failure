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
| 2 | Mini-experiment (rep99) | done — 12/17 combos clean, 5 hit a known non-blocking AUC edge case | [report](generated_data/rep99/mini_experiment_status_report.txt) |
| 2.1 | Feature-importance + calibration/decision-curve analysis (rep99) | done | reports/charts in `generated_data/rep99/` |
| 2 | PMF-fix rerun (rep99) | done | [report](generated_data/rep99/stage2_pmf_fix_rerun_report.txt) |
| 2.1 | PMF-fix rerun analyses (rep99) | done | [report](generated_data/rep99/stage2_pmf_fix_rerun_report.txt) |
| 2.2 | Debug self-check on Stage 2.1's rep99 analysis | done — Findings #1-#3 fixed, verified, closed; 2 open questions remain | [report](generated_data/rep99/stage2_2_debug_report.txt) |
| 3.0 | rep1 four_features/eight_features run (all 11 models) + analysis | done — training + analysis both regenerated, no errors (ddh/eight_features cancelled at trial 7/10 by user request, model still usable) | [report](generated_data/rep1/stage3_0_rep1_run_report.txt) |
| 3.0.1 | Debug self-check (rep1 four/eight_features) | done — no new blocking issues | [report](generated_data/rep1/stage3_0_rep1_run_report.txt) |
| 3.0.2 | rep1 twenty_features_heterogeneous run (10 models) | in progress | [log](generated_data/rep1/stage3_0_background_process_log.txt) |
| 3.0.3 | Debug self-check on combined 3-scenario analysis | not started, blocked on 3.0.2 | [plan](EXPERIMENT_PLAN_DETAILS.md) |
| 3.1 | rep2-5 full runs | rep2, rep3 (sunlab-serv-02) and rep4, rep5 (sunlab-serv-03) launched this session (gate override — see note); rep4/rep5 hit a GPU-OOM bug, fixed + relaunched — see [report](generated_data/rep4/stage3_1_rep4_rep5_gpu_oom_report.txt) | [log rep2](pkgs/scripts/eval_all_rep2.log), [log rep3](pkgs/scripts/eval_all_rep3.log), [log rep4](pkgs/scripts/eval_all_rep4.log), [log rep5](pkgs/scripts/eval_all_rep5.log) |

**Note on 3.1 gate:** Stage 3.0.2/3.0.3 (rep1 `twenty_features_heterogeneous` +
debug) had not finished/been approved when rep2/rep3 (from `sunlab-serv-02`)
and rep4/rep5 (from `sunlab-serv-03`, this row) were launched — user
explicitly instructed overriding the approval gate, disregarding the
still-running `sunlab-serv-01` process, in both sessions. All 4 of rep2-5 are
now launched, covering the full Stage 3.1 scope.

**rep4/rep5 pre-launch check (sunlab-serv-03):** before launching, verified
via `git diff`/`git status` that `pkgs/experiments/{cox,dynamic_deephit,
hazard_transformer,logistic_hazard,rnnsurv}.py` had already been edited
(uncommitted, by the user) to remove the `CKD_FIFTY_FEATURES_HETEROGENEOUS`
`os.path.exists(...)`-guarded branch from their `__main__` blocks — this
guard would otherwise have fired for rep5 only (its legacy
`ckd_fifty_features_heterogeneous_train_data.csv` already existed on disk,
rep4's didn't), retraining an unrelated out-of-scope scenario and overwriting
rep5's existing legacy model artifacts. With the guard removed, all 5
files' `__main__` blocks now only run `FOUR_FEATURES`/`EIGHT_FEATURES`/
`TWENTY_FEATURES_HETEROGENEOUS`, matching the other 6 models, so
`run_rep.sh` was used as-is for both reps with no scope creep. Also
confirmed the launching shell's Python env has numpy/torch importable and
per-experiment logs show real training output (Optuna trials, KFRE reading
real data paths) — no repeat of the rep2 `ModuleNotFoundError`-masked-success
bug noted below.

**Bug found before relaunch:** an earlier rep2 attempt (PID 12138, launched
2026-08-27, master log claimed "✓ All experiments completed successfully")
had actually failed all 11 experiments instantly with
`ModuleNotFoundError: No module named 'numpy'` (wrong/inactive Python env in
that launch's shell) — masked because `run_rep.sh`'s per-experiment subshell
always exits 0 to `wait` regardless of the inner `python -m ...`'s exit code,
so the master log's ✓/✗ line is not reliable, only each experiment's own
`eval_repN_<name>.log` is. `generated_data/rep2` and `rep3` were confirmed
empty (no scenario output ever written) before relaunching, so no work was
lost. rep3's prior attempt (PID 2460715, 2026-08-23) was from an older
sequential version of the script and had stalled after starting only `cox`.
Both were relaunched fresh this session (verified numpy/torch importable in
the launching shell first) — should be fixed in `run_rep.sh`'s exit-code
propagation before relying on its master-log ✓/✗ lines again.

## Background processes

DON'T UPDATE THIS SECTION!!!
To find background processes, run:

ps -u minhn2 -o pid,ppid,etime,%cpu,%mem,cmd --sort=-etime | grep -iE "uiuc-kidney-failure|pkgs\.(scripts|experiments|models)" | grep -v "vscode-server\|grep\|claude"

