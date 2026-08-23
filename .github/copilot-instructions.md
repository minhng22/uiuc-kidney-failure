# Repo Rules

## Plan first before executing request marked by user as "<Big Task>"
Draft plan as .md file. Plan must be approved by user.

## Background processes must be tracked in an experiment plan doc

Whenever a background process is started for this repo (training runs,
`run_all_reps.sh`, standalone rep runs, data generation, etc.), record its PID
and log file in the relevant `*_EXPERIMENT_PLAN.md` file at the repo root
(e.g. [CKD_FIFTY_FEATURES_EXPERIMENT_PLAN.md](../CKD_FIFTY_FEATURES_EXPERIMENT_PLAN.md)).

- If no experiment plan doc exists yet for the work being started, create one
  (`<EXPERIMENT_NAME>_EXPERIMENT_PLAN.md`) before/when launching the process.
- For every background process launched, note in the doc:
  - PID
  - Launch command / script
  - Log file path
  - Start time
  - What it's blocking or depends on (e.g. shared state like `current_rep` in
    `pkgs/commons.py` that must not be mutated by another concurrent run)
- Update the doc's status table/section when the process finishes, fails, or
  is found to have died — don't leave stale "in progress" / PID entries.
- Before starting a new background run, check the experiment plan doc (and
  `ps -p <pid>`) for any process that might still be running or holding
  shared state, to avoid clobbering it.

## Multiple sessions/hosts may edit these docs concurrently — don't declare another entry dead

This repo is worked on by more than one agent session at a time, sometimes on
different VMs/hosts. `ps -p <pid>` only tells you whether a PID exists on
*your* host — a miss does not mean the process is dead, only that it isn't
running on the machine you're checking from.

- When logging a background process in an `*_EXPERIMENT_PLAN.md`, include the
  hostname (`hostname`) next to the PID, since PIDs are only meaningful
  together with the host that owns them.
- Only edit/correct the status of a row for a process **you** launched (or
  otherwise confirmed is running on the host you're actually on). Don't mark
  another session's process "dead", "stalled", or "not running" based on a
  local `ps` check that doesn't match its host.
- Scope your edits to the rows/entries that are yours. If something else in
  the doc looks wrong or stale, say so in your reply to the user rather than
  rewriting it — let the owning session (or the user) correct it.

## When the user asks for a status update, update the experiment plan doc

Whenever the user asks you to check/report status on a run, don't just answer
in chat — also write the verified findings into the relevant
`*_EXPERIMENT_PLAN.md` doc (the rows/sections you own, per the rule above),
and bump its "Last Updated" timestamp. A status check that never lands in the
doc is lost the next time a session (this one or another) reads it.

## Keep EXPERIMENT_STATUS.md tidy — one report file per stage, don't swamp it with detail

`EXPERIMENT_STATUS.md` is the high-level index read at the start of every
session — it must stay skimmable end to end. **Every stage gets its own
report file** (e.g. `generated_data/rep<N>/stage<X>_<name>_report.txt`, or
an existing per-scenario report like `<scenario>_cohort_flow_report.txt`)
— that is where findings, citations, equations, coefficient tables, source
URLs, incident narratives, investigation logs, and postmortems live.
`EXPERIMENT_STATUS.md` itself never gets any of that inlined, no matter how
relevant it feels in the moment.

- A status row's "Notes" cell, and a background-process table's "Status"
  cell, hold a short status phrase plus a link to the report — e.g.
  "confirmed against primary source — see
  [report](generated_data/rep1/foo_report.txt)" or "OOM-killed, retrying —
  see [report](...)" — never a paragraph recounting the finding or telling
  the story of what happened. If you're writing more than one clause
  in `EXPERIMENT_STATUS.md`, it belongs in the report file instead.
- The "Background processes" section lists **only currently-active**
  processes (PID, host, log, one-clause status). Once a process finishes,
  move its row's detail into that stage's report and drop the row from the
  live table — don't let finished/superseded/killed runs accumulate there;
  the report file is the permanent record, not the status table.
- When updating status after doing work, put the substance in the report
  file (create one if none exists yet for that stage) and only the pointer
  + terse status in `EXPERIMENT_STATUS.md`.
- If `EXPERIMENT_STATUS.md` has accumulated inlined detail (yours or
  another session's), trimming it down to a link (moving the removed
  detail into that stage's report file, not deleting it) is fine even for
  rows you don't own — this is tidying the shared index, not editing
  another session's findings; leave the actual content intact in the
  report.

## When the user asks to add a rule, edit both instruction files

If the user asks to add/change a rule for how agents should work in this
repo, apply it to **both** [CLAUDE.md](../CLAUDE.md) and
[.github/copilot-instructions.md](copilot-instructions.md) — keep them in
sync so Claude Code and GitHub Copilot Agent Mode follow the same rules.

## Auto-check long-running operations every 10 minutes

Whenever you start (or are already watching) a long-running background
operation for this repo — a training run, `run_all_reps.sh`,
`pkgs/scripts/run_rep.sh`, data generation, etc. — set up a recurring check
every 10 minutes for as long as that operation is expected to run:

- Prefer a scheduling mechanism that survives you finishing this turn (e.g.
  a recurring/cron-style job) over a one-off manual check, so status keeps
  getting refreshed even if the user doesn't ask again.
- Each check must re-verify directly (`ps -p <pid>` on the host that owns
  it, log tail, `nvidia-smi` if GPU-bound) — never just repeat a previous
  claim without re-checking.
- Each check updates the experiment plan doc per the rule above (your own
  rows/sections, with hostname), bumping "Last Updated" — and only messages
  the user when something actually changed (stage completed, process died,
  run finished), not on every tick.
- **If a check finds that a rep you own has failed** (non-zero exit logged,
  process died with the log stalled mid-stage, etc.): diagnose the failure
  from the log/traceback, fix the underlying issue (code bug, bad path,
  resource limit, etc.), then relaunch that rep. Record the failure, the fix,
  and the relaunch (new PID) in the experiment plan doc. Only relaunch reps
  you own — never restart another session's row on a failure guess.
- Keep repeating check → (fix + relaunch if failed) → check until every rep
  you own has finished successfully, or you're told to stop. Don't stop the
  loop just because one attempt failed — a failure is a reason to fix and
  retry, not to give up.
- Stop the recurring check once all reps you own finish, or you're told to
  stop.

## Check a script's actual entry point before running it as an experiment

Before running any `pkgs/experiments/*.py` module (directly, via
`python -m`, or via a wrapper like `run_rep.sh`/`run_stage2_*.py`), read its
`if __name__ == '__main__':` block (or whatever function the invocation
actually calls) first — don't assume it only runs the scenario(s) you
intend.

- **Why:** these `__main__` blocks hardcode a specific list of scenarios,
  and that list is edited independently of whatever task you're currently
  running (e.g. Stage 1b added 3 new scenarios to `cox.py`/
  `dynamic_deephit.py`/`hazard_transformer.py`/`logistic_hazard.py`/
  `rnnsurv.py`'s existing `__main__` blocks, all of which already
  unconditionally ran `CKD_FIFTY_FEATURES_HETEROGENEOUS` first). Launching
  `pkgs/scripts/run_rep.sh 99` for the Stage 2 rep99 mini-experiment ran
  headlong into this: `CKD_FIFTY_FEATURES_HETEROGENEOUS`'s rep99 data
  didn't exist (deliberately not built — its rep1 source was mid another
  session's schema migration), so `get_train_test_data()` silently fell
  through to a full raw MIMIC extraction from `labevents.csv` instead of
  erroring — a ~2-hour, unrelated, off-scope job that looked like healthy
  CPU activity for over an hour before the real cause was found by
  inspecting the process's open file descriptors.
- If the entry point would run a scenario/experiment you don't intend,
  don't run it as-is — write a small scoped driver that imports and calls
  the underlying function(s) directly for just the scenario(s) you need
  (see [pkgs/scripts/run_stage2_new_scenarios.py](pkgs/scripts/run_stage2_new_scenarios.py)
  for the pattern), rather than editing the shared `__main__` block, unless
  the user has approved that edit (per the previous-stage-work rule above).

## When a bug is found in experiment code, verify the fix on rep99 first

If you discover a bug in `pkgs/experiments/*.py` (or code it depends on,
e.g. `pkgs/experiments/utils.py`, `pkgs/models/*.py`) — whether found while
investigating a failure, during a status check, or any other way — fix it,
then verify the fix on **rep99** before trusting it on rep1-5 or any other
full-scale rep:

- Re-run the affected model(s)/scenario(s) against rep99 data (reuse/relaunch
  the Stage 2 scoped driver pattern — `pkgs/scripts/run_stage2_new_scenarios.py`
  or equivalent — not a full rep1-5 run) and confirm the bug no longer
  reproduces there.
- Then re-run feature-importance analysis on rep99 for the affected
  scenario(s) (`pkgs/scripts/run_stage21_feature_importance.py` or
  equivalent), since a model-training fix can change which model artifacts
  exist/are valid for that analysis to consume.
- Only after both of those pass, consider the fix verified — report the bug,
  the fix, and the rep99 verification result to the user before (or as part
  of) relaunching/continuing any full-scale reps that hit the bug.
- This is in addition to, not instead of, the existing rule about full-scale
  reps you own that fail (diagnose, fix, relaunch) — rep99 verification
  comes first, full-scale relaunch comes after.
