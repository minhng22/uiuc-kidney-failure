# Feature Set Experiment Plan (4/8/20 features)

Tracks execution of [EXPERIMENT_PLAN_DETAILS.md](EXPERIMENT_PLAN_DETAILS.md) (Stage 0 plan,
approved). Do not restart another session's row without confirming its host is actually dead.

## Status

| Stage | Task | Status | Notes |
|---|---|---|---|
| 1a | Task A: determine 20 lab features | **done** | Report: [generated_data/rep1/twenty_features_lab_analysis_report.txt](generated_data/rep1/twenty_features_lab_analysis_report.txt). |
| 1a | Task B: locate Tangri et al. 8-variable KFRE coefficients | **done, fully primary-sourced, no caveats** | Report: [generated_data/rep1/kfre_8variable_coefficients_report.txt](generated_data/rep1/kfre_8variable_coefficients_report.txt). |
| — | **Stage 1a complete, zero open caveats — awaiting user approval before 1b/1c per EXPERIMENT_PLAN_DETAILS.md's explicit stop-and-wait gate.** | | |
| 1b | Code changes (commons.py, types.py, time_series_store.py, model_data_store.py, experiments/*.py, kfre.py) | **done** | types.py enum, commons.py path constants, store.py `get_uacr_df`, time_series_store.py merge logic + scenario branches, model_data_store.py `get_train_test_data` branches, all 5 experiments/*.py (cox/ddh/hazard_transformer/logistic_hazard/rnnsurv), and `pkgs/experiments/kfre.py` (closed-form 4-/8-var KFRE, coefficients per generated_data/rep1/kfre_8variable_coefficients_report.txt) — added to `run_rep.sh`'s EXPERIMENTS array. Smoke-tested against rep1 data: `four_features` C-index 0.688/AUC 0.690, `eight_features` C-index 0.637/Brier 0.396/AUC 0.663 (risk-score cache files written). |
| 1c-0 | Pilot extraction (rep1 only) + cohort-flow analysis, **approval gate** | **done, approved by user** | Table-1-style report implemented; approved to proceed to 1c. See Background processes below. |
| 1c | Data extraction (rep2-5, parallel; rep1 already done under 1c-0) | **rep2/3/4 done, rep5 retrying (OOM-killed once, relaunched)** | see Background processes below |
| 2 | Mini-experiment (rep99) | **in progress (fixed)** | scoped driver running (PID 2789348), no longer touches CKD_FIFTY_FEATURES_HETEROGENEOUS. See [report](generated_data/rep99/mini_experiment_status_report.txt). |
| 2.1 | Feature-importance analysis | not started | blocked on 2 |
| 3 | Full experiment runs (rep1-5) | not started | blocked on 2 |

## Background processes

| PID | Host | Launch command | Log | Start time | Status |
|---|---|---|---|---|---|
| ~~1027916/1027933~~ | sunlab-serv-01.cs.illinois.edu | superseded run (same-admission uACR window) | [...extraction_20260821_195128.log](pkgs/scripts/logs/pilot_rep1_new_scenarios_extraction_20260821_195128.log) | 2026-08-21 19:51 CDT | **killed** — `four_features` finished under the old design and showed severe, outcome-correlated attrition (see report); killed before `eight_features`/`twenty_features_heterogeneous` completed so the fix below could apply to all three. Stale `four_features_{train,test}_data.csv` renamed to `*.old_same_admission_uacr_bak_20260821_2012.csv`, not deleted. |
| ~~1032596~~ | sunlab-serv-01.cs.illinois.edu | `CKD_REP=1 python -c "..."` ran `get_train_test_data` for `FOUR_FEATURES`, `EIGHT_FEATURES`, `TWENTY_FEATURES_HETEROGENEOUS` in sequence, **with uACR matching loosened to whole-patient-history** (`by='subject_id'` instead of `by='hadm_id'`, per user direction — see EXPERIMENT_PLAN_DETAILS.md "1a-2" addendum) | [...extraction_v2_20260821_201233.log](pkgs/scripts/logs/pilot_rep1_new_scenarios_extraction_v2_20260821_201233.log) | 2026-08-21 20:12 CDT | **finished cleanly** (all 3 scenarios, ~2h1m total: four_features 12.3min, eight_features 34.6min, twenty_features_heterogeneous 120.4min). Process exited, nothing still running. |

Deliberately did **not** run `model_data_store.py`'s full `__main__` (which now also includes
`CKD_FIFTY_FEATURES_HETEROGENEOUS`) because `generated_data/rep1/ckd_fifty_features_heterogeneous_{train,test}_data.csv`
are currently missing/renamed to `*.old_schema_bak_20260820_192307.csv` (another session's in-progress
schema migration, dated yesterday) — invoking the full `__main__` would have auto-regenerated that
scenario's data too, an expensive, unrelated, and possibly-conflicting side effect. Scoped this pilot to
just the 3 new scenarios via a direct `get_train_test_data` call per scenario instead.

### 1c-0: closed — both approvals given 2026-08-21

Same-admission uACR bound (first pilot pass) caused severe, outcome-correlated attrition; loosened to
whole-patient-history matching per user direction (see EXPERIMENT_PLAN_DETAILS.md "1a-2" addendum),
reran clean. rep1 final results, patient-level:

| Scenario | N patients (% of source) | N records | Outcome-positive rate (source: 91.87%) |
|---|---|---|---|
| `four_features` | 2,809 (8.18%) | 26,829 | 83.52% (−8.36 pts) |
| `eight_features` | 1,213 (3.53%) | 5,407 | 85.08% (−6.80 pts) |
| `twenty_features_heterogeneous` | 32,601 (94.96%) | 8,126,090 | 91.45% (−0.42 pts) |

Reports (Table-1-style, per EXPERIMENT_PLAN_DETAILS.md "1c-0"'s approved analysis table):
[four_features](generated_data/rep1/four_features_cohort_flow_report.txt),
[eight_features](generated_data/rep1/eight_features_cohort_flow_report.txt),
[twenty_features_heterogeneous](generated_data/rep1/twenty_features_heterogeneous_cohort_flow_report.txt).

### 1c: rep2/3/4 done, rep5 retrying after an OOM kill

| PID | Host | Rep | Log | Status |
|---|---|---|---|---|
| ~~1081789~~ | sunlab-serv-01.cs.illinois.edu | 2 | [full_1c_extraction_rep2_20260821_232528.log](pkgs/scripts/logs/full_1c_extraction_rep2_20260821_232528.log) | **done** (all 3 scenarios, `ALL DONE rep2`) |
| ~~1081790~~ | sunlab-serv-01.cs.illinois.edu | 3 | [full_1c_extraction_rep3_20260821_232528.log](pkgs/scripts/logs/full_1c_extraction_rep3_20260821_232528.log) | **done** (all 3 scenarios, `ALL DONE rep3`) |
| ~~1098257~~ | sunlab-serv-01.cs.illinois.edu | 4 | [full_1c_extraction_rep4_20260822_000318.log](pkgs/scripts/logs/full_1c_extraction_rep4_20260822_000318.log) | **done** (all 3 scenarios, `ALL DONE rep4`) |
| ~~1098598~~ | sunlab-serv-01.cs.illinois.edu | 5 | [full_1c_extraction_rep5_20260822_000433.log](pkgs/scripts/logs/full_1c_extraction_rep5_20260822_000433.log) | **OOM-killed** ~00:60-01:02 CDT, confirmed via `dmesg` (`Killed process 1098598 (python)`, ~59GB resident) mid `eight_features` ESRD-positive branch. `four_features` had already completed/cached; `eight_features`/`twenty_features_heterogeneous` were not. |
| 1218942 | sunlab-serv-01.cs.illinois.edu | 5 (retry) | [full_1c_extraction_rep5_retry_20260822_101244.log](pkgs/scripts/logs/full_1c_extraction_rep5_retry_20260822_101244.log) | in progress — relaunched 2026-08-22 10:12 CDT, scoped to just the 2 missing scenarios (`EIGHT_FEATURES`, `TWENTY_FEATURES_HETEROGENEOUS`; `four_features` reused from cache). Nothing else running concurrently, no contention risk this time. |

Each runs `CKD_REP=<rep> python -c "..."` calling `get_train_test_data` for
`FOUR_FEATURES`/`EIGHT_FEATURES`/`TWENTY_FEATURES_HETEROGENEOUS` in sequence (same scoped invocation
as the 1c-0 pilot, not the shell script — still avoids touching `ckd_fifty_features_heterogeneous`'s
mid-migration rep files). Writes only to each rep's own
`generated_data/rep<N>/{four,eight,twenty}_features*_data.csv` — no shared state between reps.

**Incident 23:32 CDT (resolved)**: launching all 4 simultaneously nearly exhausted RAM+swap
(MemAvailable dropped to 1.6GB, SwapFree to 830MB — each `pd.read_csv` process was at ~46-47GB
resident and still climbing, since all 4 hit their first, heaviest labevents.csv read at the same
moment). Killed rep4/rep5 (minimal progress lost, still at the cheap `source_population` stage) to
protect rep2/rep3. Relaunched rep4 once memory recovered to 147GB available, waited ~90s and confirmed
stable, then relaunched rep5 — staggering the starts avoided a repeat synchronized spike. Verified
stable twice more (116GB available with all 4 running, each ~16-18GB, swap untouched) before leaving
them running. Lesson for any future rep launches on this host: stagger starts by a couple minutes
rather than firing all N at once, to avoid all processes' first heavy read landing simultaneously.

Separately, an unrelated environment issue was found and fixed while running `kfre.py`: an earlier
`conda install -c conda-forge poppler` (for PDF page rendering during the KFRE literature research)
had downgraded `sqlite`, breaking `optuna`'s import (used by `pkgs/experiments/utils.py`, needed by
every experiments module including `kfre.py`) — fixed via `conda install --revision 0` in the `minhn2`
env, reverting cleanly to the pre-poppler state; verified `sqlite3`/`optuna` import correctly again.

**Note for whoever picks this up next**: per the user, Stage 2 (rep99 mini-experiment, row above) is
being run concurrently from a different VM against this same repo — this session will stop once 1c
(this section) finishes; do not extend into Stage 2/2.1/3 from here.

**Status check 2026-08-22 10:12 CDT** (re-verified directly via `ps`/log tail/`dmesg`, not repeated
from prior claim): rep2/rep3/rep4 all finished cleanly (`ALL DONE` in each log, CSVs present for all
3 scenarios). rep5 was silently OOM-killed sometime after the 00:14 check — no process running, no
error in its log, confirmed via `dmesg`'s memcg OOM record for PID 1098598. `four_features` had
already cached for rep5; relaunched scoped to just `eight_features`+`twenty_features_heterogeneous`,
running alone now (no concurrent contention).

Last Updated: 2026-08-22 10:13 CDT (sunlab-serv-01.cs.illinois.edu)

### Stage 2: rep99 mini-experiment

PID 2686002 killed 01:45 CDT (was about to run a full, unrelated
CKD_FIFTY_FEATURES_HETEROGENEOUS extraction, not a hang). Fixed and
relaunched 10:12 CDT as PID 2789348, scoped to only the 3 new scenarios via
new `pkgs/scripts/run_stage2_new_scenarios.py`. Full detail in
[generated_data/rep99/mini_experiment_status_report.txt](generated_data/rep99/mini_experiment_status_report.txt).

Last Updated: 2026-08-22 10:15 CDT (sunlab-serv-02.cs.illinois.edu)
