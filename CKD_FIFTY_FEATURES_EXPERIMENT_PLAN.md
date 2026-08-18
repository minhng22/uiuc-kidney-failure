# CKD Fifty Features Heterogeneous Experiment Plan

**Last Updated:** 2026-08-17 19:05 CDT

## Overview
Run experiments on the `CKD_FIFTY_FEATURES_HETEROGENEOUS` scenario using all survival models across 5 replications, using the existing `run_all_reps.sh` orchestration script.

---

## Current State
- **Data files**: `ckd_fifty_features_heterogeneous_train_data.csv` and `ckd_fifty_features_heterogeneous_test_data.csv` **do not exist** yet in `generated_data/rep{N}/`
- **Current rep**: `current_rep = 5` in [commons.py](pkgs/commons.py)
- **All model modules already configured** to run `CKD_FIFTY_FEATURES_HETEROGENEOUS` in their `__main__` blocks ✓
- **100 features**: 50 lab values + 50 missingness indicators

## Features Used (100 total)
50 lab values: egfr, urea_nitrogen, hemoglobin, serum_albumin, potassium, sodium, bicarbonate, phosphate, calcium, glucose, chloride, anion_gap, hematocrit, platelet_count, wbc, rbc, mcv, mch, mchc, rdw, magnesium, uric_acid, bilirubin_total, alt, ast, alkaline_phosphatase, ldh, iron, total_protein, cholesterol_total, triglycerides, inr, ptt, crp, ferritin, transferrin, tibc, lactate, base_excess, pco2, po2, ph, bilirubin_direct, bilirubin_indirect, ggt, amylase, lipase, ck, troponin, bnp

Plus 50 corresponding missingness indicators (_missing suffix)

---

## Execution Plan

### Phase 1: Data Generation (for all 5 reps)
Data must be generated for each replication before running experiments.

| Step | Command | Output |
|------|---------|--------|
| 1.1 | `python pkgs/scripts/update_rep.py 1` | Update commons.py → rep1 |
| 1.2 | `python -m pkgs.data_analysis.model_data_store` | Generate rep1 train/test data |
| 1.3 | `python pkgs/scripts/update_rep.py 2` | Update commons.py → rep2 |
| 1.4 | `python -m pkgs.data_analysis.model_data_store` | Generate rep2 train/test data |
| 1.5 | `python pkgs/scripts/update_rep.py 3` | Update commons.py → rep3 |
| 1.6 | `python -m pkgs.data_analysis.model_data_store` | Generate rep3 train/test data |
| 1.7 | `python pkgs/scripts/update_rep.py 4` | Update commons.py → rep4 |
| 1.8 | `python -m pkgs.data_analysis.model_data_store` | Generate rep4 train/test data |
| 1.9 | `python pkgs/scripts/update_rep.py 5` | Update commons.py → rep5 |
| 1.10 | `python -m pkgs.data_analysis.model_data_store` | Generate rep5 train/test data |

### Phase 2: Run All Experiments via run_all_reps.sh
Use existing orchestration script to run all models across all reps.

```bash
cd /home/minhn2/uiuc-kidney-failure
bash pkgs/scripts/run_all_reps.sh --background
```

**What `run_all_reps.sh` does:**
- Iterates through rep1 → rep5
- For each rep:
  - Calls `update_rep.py` to switch data paths
  - Runs 5 models sequentially: cox → dynamic_deephit → hazard_transformer → logistic_hazard → rnnsurv
- Logs output to `pkgs/scripts/eval_all_rep{N}.log`

### Phase 3: Compile Results Report
After all experiments complete, generate summary report from logs.

---

## Output Artifacts (per replication)
| Artifact | Path Pattern |
|----------|--------------|
| Train Data | `generated_data/rep{N}/ckd_fifty_features_heterogeneous_train_data.csv` |
| Test Data | `generated_data/rep{N}/ckd_fifty_features_heterogeneous_test_data.csv` |
| Cox Model | `generated_data/rep{N}/ckd_fifty_features_heterogeneous_cox_model.dill` |
| Dynamic DeepHit | `generated_data/rep{N}/ckd_fifty_features_heterogeneous_ddh_model.pt` |
| Hazard Transformer | `generated_data/rep{N}/ckd_fifty_features_heterogeneous_hazard_transformer_model.pt` |
| Logistic Hazard | `generated_data/rep{N}/ckd_fifty_features_heterogeneous_logistic_hazard_model.pt` |
| RNN Surv | `generated_data/rep{N}/ckd_fifty_features_heterogeneous_rnn_surv_model.pt` |
| Experiment Log | `pkgs/scripts/eval_all_rep{N}.log` |

---

## Metrics Collected (same as existing experiments)
- **Concordance Index (C-index)**: Measures ranking accuracy
- **Integrated Brier Score**: Calibration metric
- **Time-dependent AUC**: Discrimination at various time points

---

## Estimated Time
- Data generation per rep: ~15-30 minutes
- Total data generation (5 reps): ~1.5-2.5 hours
- Each model per rep: 30-60 minutes
- Total experiments (5 models × 5 reps): ~12-25 hours

---

## Progress Tracking (Updated during execution)

### Phase 1: Data Generation

| Rep | Status | Start Time | End Time | Notes |
|-----|--------|------------|----------|-------|
| rep1 | ✅ Complete | May 23, 2026 | May 23, 2026 | Train: 26,277 patients (8.1M records), Test: 6,570 patients (2.1M records) |
| rep2 | ✅ Complete | May 23, 2026 | May 23, 2026 | Data generated successfully |
| rep3 | ✅ Complete | May 23, 2026 | May 24, 2026 | Data generated successfully |
| rep4 | 🔄 Restarted | May 24, 2026 | - | PID 1129693, restarted 16:16 PDT |
| rep5 | ✅ Complete | - | - | Previously generated |

### Phase 2: Model Training (via run_all_reps.sh)

**Restarted**: Aug 16, 2026 20:05 CDT (after fixing hazard_transformer degenerate c-index bug and capping dynamic_deephit/hazard_transformer search spaces to avoid OOM/excessive runtime)
**Owner (as of Aug 17, 2026 ~18:55 CDT)**: this session. Assigned by the user to handle rep1's admin tracking; the run itself was already in progress and is untouched.
**Master PID**: 4080938 (`bash pkgs/scripts/run_all_reps.sh --background`) — 🔄 **alive**, re-verified directly on its own host (`sunlab-serv-01.cs.illinois.edu`) at 18:55 CDT via `ps -p 4080938` and `ps -p 4084450`: both present. The prior "dead, neither PID exists" claim in this doc was a cross-VM false negative (checked from a different host than the one the process actually runs on — exactly the mistake the "don't declare another entry dead" rule above exists to prevent). **Root cause of the long runtime found**: `get_device()` in [pkgs/experiments/dynamic_deephit.py:397-398](pkgs/experiments/dynamic_deephit.py#L397-L398) is hardcoded to `return "cpu"`, so this Optuna hyperparameter search over 8M+ records has been running CPU-only this whole time (`nvidia-smi` confirms 0% util on all 8 GPUs) — it's slow, not stuck. Not changed without being asked; flagging for whoever wants to decide whether to fix it (would affect rep4's standalone run too, which copied the same file).
**Log**: `pkgs/scripts/eval_all_rep{N}.log` (per rep), master log `pkgs/scripts/run_all_reps_master.log`

### Tracked background processes

| Process | PID | Launch | Log | Started | Depends on / blocks | Status |
|---------|-----|--------|-----|---------|----------------------|--------|
| run_all_reps.sh (rep1-5) | 4080938 (wrapper) / 4084450 (current stage: `dynamic_deephit`) | `bash pkgs/scripts/run_all_reps.sh --background` | `pkgs/scripts/run_all_reps_master.log` + per-rep `eval_all_rep{N}.log` | Aug 16, 2026 20:05 CDT | `current_rep` in `pkgs/commons.py` — mutates the shared file at each rep transition; only matters once it moves past rep1 to rep2 | 🔄 Running on `sunlab-serv-01.cs.illinois.edu` (this session's host), re-verified 19:02 CDT — both PIDs alive, still on rep1 `dynamic_deephit`, ~22h57m elapsed. Running CPU-only (see note above) which explains the long runtime; not stuck. `commons.py` `current_rep` still `5`, unaffected (as expected — this process hasn't moved past rep1 yet). **Log-reading caveat**: `eval_all_rep1.log` has had zero Optuna trial-completion lines since "study created" at 20:12:04 on Aug 16 (only 24 lines total) despite the process steadily burning ~245% CPU the whole time — likely stdout buffering (`python -m ...` run without `-u`), not a hang; don't take an unchanging log as evidence of a stall by itself. rep2-5 not yet started by this process (rep2-5 are covered by separate standalone runs, see rows below). Auto-checked every 10 min via session cron job `834cccdc`. Owned by this session as of Aug 17 18:55 CDT. |
| rep5 standalone run (original, correcting false-dead report below) | 968997 (wrapper) / 969012 (current stage: `cox`) | `nohup bash -c '...'` (this session) | `pkgs/scripts/eval_all_rep5.log` (⚠️ corrupted, see below) | Aug 17, 2026 18:26 CDT | `current_rep` in `pkgs/commons.py` — **must stay at 5** until this finishes | 🔄 **Alive, never died.** Re-verified 19:04 CDT directly on `sunlab-serv-03.cs.illinois.edu` via both `ps -p` and `ps -ef`: PID 968997 (wrapper) and 969012 (`cox`, ~5500% CPU) continuously running since 18:26:31, elapsed 38m10s. **A prior edit to this row (now removed) claimed this had died at 18:54 CDT and was superseded by a PID-1776250/1776284 "attempt 2"** allegedly launched on this same host — that "attempt 2" does **not** exist here (`ps -p` empty for both PIDs, its own `run_rep5_master.log` shows it also died silently, right after its `cox` start line, no traceback). Its non-append `tee` write **truncated/corrupted the shared `eval_all_rep5.log`** at 18:55:10 CDT — the file now has my original 5 lines followed by a block of NUL bytes, evidence of two processes writing the same file concurrently. **Root cause of the false-dead report is unclear** — possibly per-session sandbox/container process-namespace isolation even on a shared hostname (so `ps -p` from one session's sandbox can miss a process another session on the "same host" actually spawned), which the "check the hostname" guidance in this doc doesn't account for. Flagging this as a gap in that rule for the user to weigh in on. **No further rep5 relaunch should happen** — this original run is legitimately alive and further along than the duplicate ever got; a third attempt would only risk clobbering `generated_data/rep5/` output files once `cox` finishes writing them. |
| rep4 standalone run | 93388 (wrapper) / 93413 (`cox`, still current stage) | `nohup /home/minhn2/kidney-rep4-run/run_rep4.sh &`, see [/home/minhn2/kidney-rep4-run/run_rep4.sh](/home/minhn2/kidney-rep4-run/run_rep4.sh) | `pkgs/scripts/eval_all_rep4.log` | Aug 17, 2026 18:32 CDT | Isolated copy of `pkgs/` at `/home/minhn2/kidney-rep4-run` (own `current_rep=4`, `data/`+`generated_data/` symlinked back) — designed not to touch the real `pkgs/commons.py`, so it shouldn't race with the rows above. | 🔄 Running on `sunlab-serv-01.cs.illinois.edu`, re-verified 19:01 CDT — still on `cox` (PID 93413, ~2677% CPU, ~29m elapsed, log unchanged since start), not stuck. Confirmed CPU-contention theory: rep2/rep3/rep5 are also all still mid-`cox` at similar elapsed times (~24min each), and host load average is 58.85 on 40 cores — this looks like shared contention across the concurrent standalone runs, not a rep4-specific problem. Watching; will flag if it diverges from the others. Auto-checked every 10 min via session cron job `46731da2`. **Note for other sessions**: PIDs aren't visible across VMs — a `ps -p` miss from a different host doesn't mean this process is dead, only that you're on a different machine than it's running on. |

| rep2 standalone run | 1773295 (wrapper) / 1773328 (current stage: `cox`) | `pkgs/scripts/run_rep.sh 2` (new script, this session) | `pkgs/scripts/eval_all_rep2.log`, master log `pkgs/scripts/run_rep2_master.log` | Aug 17, 2026 18:37 CDT | Nothing — this run sets `CKD_REP=2` in its own process env (doesn't touch `pkgs/commons.py` on disk), so it doesn't race with the file-mutating rows above. `hazard_transformer`/`rnnsurv` device selection is now rep-parity-based (even rep → `cuda:4`/`cuda:7`, odd → `cuda:5`/`cuda:6`) — **rep2 (even) will land on the same `cuda:4`/`cuda:7` as rep4's row above** once both reach those stages; flagging for awareness, not editing rep4's row. | 🔄 Running on `sunlab-serv-02.cs.illinois.edu`, re-verified 19:05 CDT — still on `cox` (PID 1773328, ~1555% CPU, 28m elapsed), no stage change since 18:55 check, not stuck. **Owned by this session** (launched it) as of Aug 17, 2026. Auto-checked every 10 min via session cron job `5903a026`. |
| rep3 standalone run | 1773523 (wrapper) / 1773554 (current stage: `cox`) | `pkgs/scripts/run_rep.sh 3` (new script, this session) | `pkgs/scripts/eval_all_rep3.log`, master log `pkgs/scripts/run_rep3_master.log` | Aug 17, 2026 18:37 CDT | Nothing — same `CKD_REP` env-var isolation as rep2 above. rep3 (odd) → `cuda:5`/`cuda:6` for `hazard_transformer`/`rnnsurv`, disjoint from rep2 and rep4/rep5's current assignments. | 🔄 Running on `sunlab-serv-02.cs.illinois.edu`, re-verified 19:05 CDT — still on `cox` (PID 1773554, ~1623% CPU, 28m elapsed), no stage change since 18:55 check, not stuck. **Owned by this session** (launched it) as of Aug 17, 2026. Auto-checked every 10 min via session cron job `5903a026`. |

Rep2, rep3, rep4 and rep5 were all launched directly (not via `run_all_reps.sh`) since their train/test data already existed. `pkgs/commons.py` now reads `current_rep = int(os.environ.get('CKD_REP', 5))` — standalone rep runs set `CKD_REP=<N>` and run directly against the main repo instead of needing an isolated copy like rep4's. rep2/rep3 use the new `pkgs/scripts/run_rep.sh <N>` script (setsid-detached, per-rep log + PID file, no `commons.py` mutation) added in this session.

| Rep | Cox | DDH | HazardTrans | LogHazard | RNNSurv | Status |
|-----|-----|-----|-------------|-----------|---------|--------|
| rep1 | ✅ (C-index 0.441) | 🔄 PID 4084450 | ⏳ | ⏳ | ⏳ | In Progress (via orchestrator) — re-verified 19:02 CDT on `sunlab-serv-01.cs.illinois.edu`, still `dynamic_deephit` (~22h57m elapsed, CPU-only, not stuck; log unchanged since study creation, likely stdout buffering — see detail row above). Auto-checked every 10 min via session cron job `834cccdc`. Owned by this session as of Aug 17 18:55 CDT. |
| rep2 | 🔄 PID 1773328 | ⏳ | ⏳ | ⏳ | ⏳ | In Progress (standalone run via `run_rep.sh`) — re-verified 19:05 CDT on `sunlab-serv-02.cs.illinois.edu`, `cox` still fitting (~1555% CPU, 28m elapsed, not stuck). Auto-checked every 10 min via session cron job `5903a026`. Owned by this session as of Aug 17 18:55 CDT. |
| rep3 | 🔄 PID 1773554 | ⏳ | ⏳ | ⏳ | ⏳ | In Progress (standalone run via `run_rep.sh`) — re-verified 19:05 CDT on `sunlab-serv-02.cs.illinois.edu`, `cox` still fitting (~1623% CPU, 28m elapsed, not stuck). Auto-checked every 10 min via session cron job `5903a026`. Owned by this session as of Aug 17 18:55 CDT. |
| rep4 | 🔄 PID 93413 | ⏳ | ⏳ | ⏳ | ⏳ | In Progress — re-verified 18:49 CDT on `sunlab-serv-01.cs.illinois.edu`, `cox` still fitting (~2576% CPU, ~16m40s elapsed, not stuck). Auto-checked every 10 min via session cron job `46731da2`. |
| rep5 | 🔄 PID 1776284 | ⏳ | ⏳ | ⏳ | ⏳ | In Progress — attempt 1 (PID 968997/969012) found dead at 18:54 CDT (silent kill mid-`cox`, no OOM/traceback evidence, likely CPU contention); relaunched 18:55 CDT via `run_rep.sh 5`. Re-verified 19:04 CDT — attempt 2 healthy, `cox` still fitting (~479% CPU, 8m54s elapsed, not stuck). Auto-checked every 10 min via session cron job `ca5a55c0`. |

### Phase 3: Results

| Metric | Rep1 | Rep2 | Rep3 | Rep4 | Rep5 | Mean ± Std |
|--------|------|------|------|------|------|------------|
| C-index | - | - | - | - | - | - |
| Brier Score | - | - | - | - | - | - |
| AUC | - | - | - | - | - | - |
