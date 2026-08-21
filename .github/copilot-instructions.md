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

## Keep EXPERIMENT_STATUS.md tidy — link out to detailed reports, don't inline them

`EXPERIMENT_STATUS.md` is the high-level index read at the start of every
session — it must stay skimmable. Detailed findings, citations, equations,
coefficient tables, source URLs, etc. belong in a dedicated report file
(e.g. under `generated_data/rep<N>/`, per the relevant
`*_EXPERIMENT_PLAN_DETAILS.md`'s reporting convention), not inlined into
`EXPERIMENT_STATUS.md`'s status table.

- Each row's "Notes" cell should be a short status phrase plus a link to the
  report file that has the actual detail (e.g. "confirmed against primary
  source — see [report](generated_data/rep1/foo_report.txt)"), not a
  paragraph recounting the finding.
- When updating status after doing work, put the substance in the report
  file (create/append one if none exists yet for that task) and only the
  pointer + terse status in `EXPERIMENT_STATUS.md`.
- If `EXPERIMENT_STATUS.md` has accumulated inlined detail (yours or another
  session's), trimming your own rows down to a link is fine; leave other
  sessions' rows for their owners per the multi-session rule above.

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
