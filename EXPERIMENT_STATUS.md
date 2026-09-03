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

| What | PID | Host | Log | Status |
|---|---|---|---|---|
| Stage 3.0.2 rep1 twenty_features_heterogeneous (10 models) | -- (leader 3131354 exited) | sunlab-serv-01.cs.illinois.edu | [log](generated_data/rep1/stage3_0_background_process_log.txt) | 9/10 done (cox/logistic_hazard/deepsurv/srf/hazard_transformer/survival_svm/weibul/rnnsurv all valid); gbsa still running (PID 3466503, 1d15h+, slow but memory-stable); dynamic_deephit relaunched per user request (PID 3903075), running in parallel -- same code, expect similarly long runtime (28h+ last attempt), see log; a real rnnsurv eval-time bug found (_batched_risk_by_time unbatched concat) -- not fixed, has a valid result via a lucky concurrent run, see log |
| Stage 3.1 rep2 full run (11 models, four/eight/twenty scenarios) | 531632 | sunlab-serv-02.cs.illinois.edu | [master log](pkgs/scripts/run_rep2_master.log), per-exp `pkgs/scripts/eval_rep2_<name>.log` | **8/11 done** (hazard_transformer finished successfully — C-index 0.7519, AUC 0.85, Brier 0.462); **srf notably still alive** (PID 1492481, ~10h wall-clock now, RSS only ~6GB — far longer/lighter than any prior attempt, likely benefiting from hazard_transformer having freed up memory) — not touched, letting it run; 22+ prior attempts all crashed before ever reaching the four/eight/twenty scenarios (still stuck in the initial NON_TIME_VARIANT GridSearchCV each time) — see [report](generated_data/rep2/stage3_1_srf_oom_report.txt); cox/rnnsurv resolved earlier — see [rnnsurv report](generated_data/rep2/stage3_1_rnnsurv_riskbytime_bug_report.txt) and [cox report](generated_data/rep2/stage3_1_cox_oom_report.txt); dynamic_deephit, gbsa still training, no new failures |
| Stage 3.1 rep3 full run (11 models, four/eight/twenty scenarios) | 532683 | sunlab-serv-02.cs.illinois.edu | [master log](pkgs/scripts/run_rep3_master.log), per-exp `pkgs/scripts/eval_rep3_<name>.log` | **8/11 done** (hazard_transformer, cox, rnnsurv, kfre, survival_svm, weibul, deepsurv, logistic_hazard); srf OOM-killed again (PID 1492618 at 23:59, anon-rss ~85.7GB), relaunched as PID 1595639 (22+ prior attempts, zero successes) — see [report](generated_data/rep2/stage3_1_srf_oom_report.txt); same cox/rnnsurv incident as rep2, resolved — see rep2's row + reports; dynamic_deephit, gbsa still training, no new failures |
| Stage 3.1 rep4 full run (11 models, four/eight/twenty scenarios via `pkgs/scripts/run_rep.sh 4`) | 332134 | sunlab-serv-03.cs.illinois.edu | [master log](pkgs/scripts/run_rep4_master.log), per-exp `pkgs/scripts/eval_rep4_<name>.log` | **9/11 done** (kfre, survival_svm, weibul, deepsurv, logistic_hazard, cox, srf, hazard_transformer, rnnsurv — relaunch succeeded, C-index 0.611, confirming Incident #3 fix works); dynamic_deephit, gbsa still actively computing — see [report](generated_data/rep4/stage3_1_rep4_rep5_gpu_oom_report.txt) |
| Stage 3.1 rep5 full run (11 models, four/eight/twenty scenarios via `pkgs/scripts/run_rep.sh 5`) | 333435 | sunlab-serv-03.cs.illinois.edu | [master log](pkgs/scripts/run_rep5_master.log), per-exp `pkgs/scripts/eval_rep5_<name>.log` | **9/11 done** (kfre, survival_svm, weibul, deepsurv, rnnsurv, logistic_hazard, cox, srf, hazard_transformer); dynamic_deephit, gbsa still actively computing — see [report](generated_data/rep4/stage3_1_rep4_rep5_gpu_oom_report.txt) |

Last Updated: 2026-09-03 09:44 CDT (sunlab-serv-02.cs.illinois.edu, rep2/rep3
rows only — rep2's srf notably alive ~10h (RSS only ~6GB, likely benefiting
from hazard_transformer having freed memory), left running untouched;
rep3's srf OOM-killed again (~85.7GB) and relaunched (PID 1595639); both
still 8/11; dynamic_deephit/gbsa confirmed alive on both reps; rep4/rep5
rows are sunlab-serv-03's own, 3.0.2 row/timestamp is sunlab-serv-01's own,
both left as-is).
Full
history for this stage (launches, incidents, health checks) is in
[stage3_0_rep1_run_report.txt](generated_data/rep1/stage3_0_rep1_run_report.txt)
and [stage3_0_background_process_log.txt](generated_data/rep1/stage3_0_background_process_log.txt);
Stage 2.2's Findings #1-#3 are in
[stage2_2_debug_report.txt](generated_data/rep99/stage2_2_debug_report.txt).
