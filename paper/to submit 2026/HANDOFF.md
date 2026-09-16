# Handoff: picking up the "to submit 2026" paper

Written for: any coding agent resuming this work — Claude Code, a GCP-hosted
agent, OpenAI Codex CLI, or a human. Nothing below is tool-specific; every
step is plain `bash`/`python`/`git`.

## What this is

A benchmark paper comparing 10 survival models + the closed-form Kidney
Failure Risk Equation (KFRE) across three MIMIC-IV feature-set scenarios
(`four_features`, `eight_features`, `twenty_features_heterogeneous`). Same
dataset/cohort lineage as `paper/submitted 2025`; different experiments and
models. Full experiment design lives at repo root:
[EXPERIMENT_PLAN.md](../../EXPERIMENT_PLAN.md) (don't edit — locked),
[EXPERIMENT_PLAN_DETAILS.md](../../EXPERIMENT_PLAN_DETAILS.md) (the actual
execution plan, read this first), [EXPERIMENT_STATUS.md](../../EXPERIMENT_STATUS.md)
(live status — check this fresh, don't trust anything below past its
"last known" framing).

**Read [CLAUDE.md](../../CLAUDE.md) at repo root before touching any
background process or experiment file.** This repo is worked on by multiple
agent sessions concurrently, sometimes on different hosts
(`sunlab-serv-01/02/03.cs.illinois.edu` seen so far). Rules that matter most
for paper work specifically: don't mark another session's row dead from a
local `ps` check; verify a model's actual code before describing its
architecture (comments lie); when you fix a bug, verify on rep99 before
touching rep1-5.

## Files in this directory

- `paper content/ml4h2026.tex` — **ML4H 2026 Proceedings-track** version.
  Uses the `jmlr` class (`\documentclass[pmlr,twocolumn,10pt]{jmlr}`),
  double-blind anonymized (`\author{Anonymous Author(s)}`, no institution/
  repo links anywhere — grep for identifying strings before editing this
  one). 5 pages of main content + refs + a 3-page appendix (well under
  ML4H's 8-page cap excl. refs/appendix). Sections:
  `sections/ml4h_introduction.tex`, `ml4h_methods.tex`, `ml4h_results.tex`,
  `ml4h_discussion.tex`, `ml4h_appendix.tex`.
- `paper content/sn-article.tex` — fuller **Springer Nature (`sn-jnl`
  class)** version, single-column, real author names, much more detail
  (full per-model architecture equations in the body, not pushed to an
  appendix). Sections: `sections/introduction.tex`, `methods.tex`,
  `results.tex`, `discussion.tex`. This is the fallback-venue base to adapt.
- `paper content/sn-bibliography.bib` — shared by both versions.
- `paper content/figs/` — PNGs pulled from `generated_data/rep1/*.png`
  (comparison charts, feature-importance grids, calibration/decision-curve
  plots). Re-copy from `generated_data/rep<N>/` if regenerating for a
  different rep.
- `drafts/` — compiled PDFs, named `<PaperTag>_<mmddhhmm><TZ>.pdf`. Keep
  producing new timestamped files here rather than overwriting; don't
  delete old ones without asking.

## Compiling (no system LaTeX installed — use tectonic)

```bash
mkdir -p /tmp/texbin && cd /tmp/texbin
curl -sL -o t.tar.gz "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz"
tar xzf t.tar.gz && chmod +x tectonic
cd "/home/minhn2/uiuc-kidney-failure/paper/to submit 2026/paper content"
/tmp/texbin/tectonic -X compile ml4h2026.tex   # or sn-article.tex
# then copy the resulting .pdf into ../drafts/ with a fresh timestamp,
# and delete the .aux/.log/.bbl/.blg/.out/.pdf build litter from this dir
```
Needs outbound network access the first run (tectonic fetches its TeX
package bundle on demand and caches it).

## Regenerating results numbers across reps

`pkgs/scripts/aggregate_rep_metrics.py` (repo root) parses the
"Discrimination metrics" block out of every
`generated_data/rep<N>/<scenario>_clinical_validity_report.txt` that
exists (tolerant of partial completion — reports which reps went into each
number) and prints mean±SD per model:

```bash
cd /home/minhn2/uiuc-kidney-failure
PYTHONPATH=. python3 -m pkgs.scripts.aggregate_rep_metrics four_features eight_features twenty_features_heterogeneous
```

Drop the output straight into `ml4h_results.tex` / `results.tex`'s
discrimination tables (they're already formatted for mean±SD, single-rep
values are just a placeholder until more reps land) and update the
footnote/caveat sentence about repetition count accordingly — don't leave
stale "rep1 only" language once more reps are in.

## Data status (last known — re-check `EXPERIMENT_STATUS.md` and re-run
the aggregator above before trusting this)

- rep1: all three scenarios trained, analyzed, complete.
- rep2/rep3/rep4: `four_features`/`eight_features` clinical-validity
  reports existed as of 2026-09-16; `twenty_features_heterogeneous` and a
  `srf` (Random Survival Forest) OOM-crash loop on rep2/rep3 were open
  problems as of the last full status check (20+ failed relaunches, 0
  successes at last look) — confirm whether that's since been fixed before
  assuming rep2/rep3 are complete.
- rep5: all three scenarios' clinical-validity reports existed as of
  2026-09-16.
- Open, disclosed-not-fixed model-behavior questions (see
  `generated_data/rep99/stage2_2_debug_report.txt` and
  `generated_data/rep1/stage3_0_rep1_run_report.txt`): a Cox model scoring
  below the 0.5 random baseline on every scenario, root cause unknown.
  Report this honestly in the paper — don't quietly drop or fabricate an
  explanation for it.

## Target venue

**ML4H 2026 Proceedings track** was the primary target — submission
deadline **was Sept 10, 2026, 11:59 PM AoE**, which has already passed as
of this doc being written (2026-09-16). Confirm with the user whether a
submission actually went in before that deadline; if not, treat `ml4h2026.tex`
as ready-to-adapt rather than as a live submission.

Fallback ranking discussed with the user (easiest-acceptance-adjusted, not
pure prestige), in order: **PLOS Digital Health** (rolling, soundness-based
review, easiest realistic bar) → **Journal of Biomedical Informatics**
(rolling, accommodating) → ML4H Proceedings itself if a later cycle/extension
applies → **IEEE JBHI** → **JAMIA** → **npj Digital Medicine** (hardest,
highest prestige/IF).

**IEEE JBHI needs a from-scratch reformat, not built yet**: `IEEEtran`
class (not `jmlr`), single-blind (real author names/affiliations/ORCID
required, do NOT anonymize), IEEE numeric bracket citation style (not
`natbib`), 14-page limit including supplementary material (roomier than
ML4H's 8 — worth restoring `sn-article.tex`'s fuller per-model detail into
the main body rather than reusing the compressed ML4H version), plus
portal-level requirements outside the PDF itself (cover letter, signed
Author Consent Form, 4 suggested reviewers, IRB reference number).
PLOS Digital Health and JBI are both closer to `sn-article.tex`'s
non-anonymized, fuller-detail shape than to `ml4h2026.tex`'s anonymized,
compressed one — start from `sn-article.tex` for any of the journal targets.
