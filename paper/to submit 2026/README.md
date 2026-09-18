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

## Which version is the live one

**`paper content/plos_digital_health.tex` is the latest paper — treat it as
the lead version.** `sn-article.tex` and `ml4h2026.tex` are the other two
venue cuts, kept compiling but no longer the place new work lands first.

The practical consequence: PLOS is a **single self-contained file** that
duplicates section prose rather than `\input`-ing `sections/*.tex`, while the
other two versions are assembled from those shared section files. So there is
no single source of truth across all three, and edits do not propagate in
either direction on their own. Whichever file you edit, hand-mirror the change
into the others and recompile all three before calling the work done.

This has already bitten once: the 2026-09-18 results update landed in
`sections/*.tex` (and so in ML4H and SN) and in PLOS's *abstract*, but PLOS's
body kept the superseded rep1/pilot text until it was hand-synced.

## Files in this directory

- `paper content/plos_digital_health.tex` — **the lead version** (see above);
  details under its bullet below.
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
  `results.tex`, `discussion.tex`. Shares those section files with
  `ml4h2026.tex`; it was the original base PLOS was adapted from, but PLOS
  has since become the lead version.
- `paper content/plos_digital_health.tex` — **PLOS Digital Health** version,
  **the latest/lead paper**, built from the official PLOS LaTeX template
  (`documentclass[10pt,letterpaper]{article}`
  with PLOS's own geometry/packages, not `sn-jnl`). Single self-contained file
  (PLOS requires one `.tex`, no `\input`), originally adapted from
  `sn-article.tex`'s
  section content: unnumbered sections in PLOS's order (Abstract, Author
  summary, Introduction, Materials and methods, Results, Discussion,
  Supporting information, Acknowledgments, References), internal
  `Section~\ref` cross-references converted to `\nameref` (sections are
  unnumbered per PLOS style), `\citep`→`\cite` (Vancouver numeric via `cite`
  package), figure/table captions kept inline but all `\includegraphics`
  stripped — PLOS requires figures uploaded as separate files, not embedded
  in the manuscript PDF. Real author names (non-anonymized, like
  `sn-article.tex`). Uses `paper content/plos2025.bst` (official PLOS
  BibTeX style) against the shared `sn-bibliography.bib`. Verified to
  compile cleanly with tectonic (no undefined refs/citations); pre-existing
  `sn-bibliography.bib` gaps surfaced by BibTeX (missing `journal` field on
  `ishwaran2008random`, missing `author`/`publisher` on `hu2022locf_bias`,
  conflicting `volume`/`number` on `lee2018deephit`) are latent issues in
  the shared `.bib`, not introduced by this file — worth fixing before
  actual submission but left untouched here since the file is shared with
  the other two venue versions.

  Because this file duplicates section prose rather than `\input`-ing it,
  changes here do not reach `ml4h2026.tex`/`sn-article.tex` and vice versa —
  see "Which version is the live one" above before editing either side.
- `paper content/sn-bibliography.bib` — shared by all three versions.
- `paper content/figs/` — PNGs pulled from `generated_data/rep1/*.png`
  (comparison charts, feature-importance grids, calibration/decision-curve
  plots). Re-copy from `generated_data/rep<N>/` if regenerating for a
  different rep. Note that the current text references only
  `four_features_all_models_feature_importance.png` and
  `four_features_calibration_plot.png` (plus the eight-feature importance
  grid as PLOS's S1 Fig); the decision-curve plots and the c-index
  comparison chart are deliberately **not** referenced any more (net-benefit
  claims were withdrawn), and the two
  `twenty_features_heterogeneous_*` PNGs added on 2026-09-18 are not yet
  cited by any version.
- `drafts/` — compiled PDFs, named `<PaperTag>_<mmddhhmm><TZ>.pdf`. Keep
  producing new timestamped files here rather than overwriting; don't
  delete old ones without asking.

## Compiling (no system LaTeX installed — use tectonic)

```bash
mkdir -p /tmp/texbin && cd /tmp/texbin
curl -sL -o t.tar.gz "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz"
tar xzf t.tar.gz && chmod +x tectonic
cd "/home/minhn2/uiuc-kidney-failure/paper/to submit 2026/paper content"
/tmp/texbin/tectonic -X compile ml4h2026.tex   # or sn-article.tex, or plos_digital_health.tex
# then copy the resulting .pdf into ../drafts/ with a fresh timestamp,
# and delete the .aux/.log/.bbl/.blg/.out/.pdf build litter from this dir
```
Needs outbound network access the first run (tectonic fetches its TeX
package bundle on demand and caches it).

On an Apple-silicon Mac, swap the release asset for
`tectonic-0.15.0-aarch64-apple-darwin.tar.gz` and use the repo path under
`~/Documents/Code/uiuc-kidney-failure`; everything else is identical. All
three versions compile with zero undefined refs/citations. BibTeX still
prints the shared-`.bib` warnings noted above (and one `ishwaran2008random`
missing-`journal` error) — non-fatal, the bibliography still builds.

## Target venue

| Venue | Impact Factor | Deadline | Est. acceptance probability |
| --- | --- | --- | --- |
| PLOS Digital Health | ~7.7 (one source: 8.5) | Rolling — none | High — stated editorial policy reviews for soundness, not novelty |
| Communications Medicine (Nature Portfolio) | 7.4 | Rolling — none | Medium-high — inferred: "Communications" tier reviews for soundness over flagship-level novelty; no sourced rate for this title |
| JMIR (Journal of Medical Internet Research) | 8.2 (5-yr 8.1) | Rolling — none | Medium-high — inferred: strong topical fit, but IF roughly doubled recently (~5-6 → 8.2), likely more competitive now; no sourced rate |

## Known content gaps (rejection risks)

Reviewer-lens read of `paper content/sections/{introduction,methods,results,discussion}.tex`
(shared by `sn-article.tex` and `plos_digital_health.tex`), ranked by how
likely each is to sink the paper at a soundness-focused venue like PLOS
Digital Health — verified against the section files, not assumed. Fix
candidates before submitting anywhere, not just at PLOS.

**Updated 2026-09-18, after the completed rep1--5 results landed.** All three
scenarios now have five completed runs (160 model--scenario--run records;
see `results/performance_summary.csv`), the twenty-feature scenario reports
the full 32,601-patient cohort rather than the n=20 pilot, and all three
venue files were rewritten around those numbers. That closes the two
completeness gaps below and reframes the rest.

### Near-certain rejection/major-revision triggers

- **3.** Uncertainty quantification is still absent in the sense reviewers
  will mean it. The tables now carry a sample SD, but it is variation across
  five refits on **one fixed patient partition** — not patient-sampling
  uncertainty. There are still no bootstrap CIs, no paired significance
  tests, and no external cohort, while comparative rankings are stated
  throughout. The sections disclose this explicitly; a reviewer may still
  require actual intervals.
- **4.** Model-selection leakage: LogisticHazard uses the designated **test**
  partition as `val_data` and for trial/checkpoint selection, so its numbers
  are not a held-out estimate. Now disclosed in Methods/Limitations rather
  than fixed — a soundness-focused venue may treat disclosure as
  insufficient.
- **5.** Evaluation-unit mismatches inside the metrics themselves: the IPCW
  censoring reference for IBS/AUC is built from **raw training rows** while
  predictions are per patient, and the reported IBS scores a shared
  scalar-risk→survival transformation rather than each model's native
  survival distribution. Decision-curve superiority claims have been
  withdrawn for the same reason (model curves use patients, comparators use
  rows), which removes what used to be the paper's clearest actionable
  finding.

### High

- **6.** Population mismatch with KFRE's validation domain. The Limitations
  now say performance "should not be generalized to routine clinical
  screening" ([discussion.tex:11](paper%20content/sections/discussion.tex#L11)),
  but Methods still frame the 83--92% ESRD rate only as "event-rich by
  construction" ([methods.tex:26](paper%20content/sections/methods.tex#L26)),
  never as external-validity spectrum bias against KFRE's outpatient
  nephrology-referral derivation setting.
- **7.** No external validation cohort — all numbers come from one internal
  train/test split, while the paper benchmarks against an equation
  validated across 31 independent external cohorts
  ([introduction.tex:3](paper%20content/sections/introduction.tex#L3)).
- **8.** Prediction time is not a prospective landmark: nearest-value matching
  can attach measurements taken *after* a row's creatinine anchor, and
  durations are never reset at a decision time. Now stated plainly in
  Limitations ([discussion.tex:13](paper%20content/sections/discussion.tex#L13)),
  which makes it easy for a reviewer to find and object to.
- **9.** No TRIPOD/TRIPOD-AI checklist for this multivariable clinical
  prediction-model study; STROBE is cited instead, for cohort flow only
  ([methods.tex:26](paper%20content/sections/methods.tex#L26)).

### Moderate

- **10.** Competing-risks handling (death before ESRD) is disclosed as
  out-of-scope ([discussion.tex:15](paper%20content/sections/discussion.tex#L15)),
  but this is often a reviewer-mandated correction in ESRD survival
  modeling specifically — disclosure may not be enough.
- **11.** Feature-importance reporting is now heavily qualified rather than
  over-claimed (cross-model ranking claims were dropped), but the
  twenty-feature Dynamic-DeepHit and Hazard Transformer gradient analyses
  **failed memory allocation in all five runs**, so that scenario has no
  usable importance results at all
  ([results.tex:98](paper%20content/sections/results.tex#L98)).
- **12.** No demographic subgroup or fairness analysis (race/sex/age strata) —
  now named as a limitation
  ([discussion.tex:15](paper%20content/sections/discussion.tex#L15)) but still
  not performed; notable gap for a "digital health" venue.
- **13.** Ethics declaration (in `sn-article.tex`'s backmatter) lacks the
  CITI-training/PhysioNet-credentialing language PLOS commonly expects
  for MIMIC-based submissions specifically.
