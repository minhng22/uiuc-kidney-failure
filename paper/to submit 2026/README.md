# Handoff: picking up the "to submit 2026" paper

Written for: any coding agent resuming this work — Claude Code, a GCP-hosted
agent, OpenAI Codex CLI, or a human. Nothing below is tool-specific; every
step is plain `bash`/`python`/`git`.

## What this is

A benchmark paper comparing 10 survival models + the closed-form Kidney
Failure Risk Equation (KFRE) across three MIMIC-IV v2.2 feature-set scenarios
(`four_features`, `eight_features`, `twenty_features_heterogeneous`). Each
scenario is split per patient 64/16/20 into train / test / internal holdout
(`<scenario>_external_validation_data.csv`) with a different seed per rep
(reps 1-5). The paper reports **holdout** performance: mean (SD) across the five
splits, patient-bootstrap 95% intervals, paired differences vs KFRE,
calibration, patient-level decision curves, and subgroup (age/sex/race)
performance. Reporting follows TRIPOD+AI (S1 Checklist).

**Read [CLAUDE.md](../../CLAUDE.md) at repo root before touching any
background process or experiment file.** This repo is worked on by multiple
agent sessions concurrently, sometimes on different hosts.

## STAND-IN NUMBERS -- replace before submission

The 2026-10-05 update was written under two explicit assumptions from the
user. Both are flagged in a comment block at the top of every `.tex` and in
`results/performance_provenance.json`:

1. **Holdout = test stand-in.** No `--external-validation` analysis has been
   run yet. Every "holdout" number and figure is currently copied from the
   test-set reports/PNGs. To replace:
   `python -m pkgs.scripts.run_experiments analyze --reps all --external-validation`
   (writes `generated_data/rep_<N>/external_validation_*`), then set
   `EVAL_REPORT_PREFIX = "external_validation_"` in `scripts/aggregate_results.py`
   and re-copy the figures from `generated_data/rep_1/external_validation_*.png`.
2. **Synthetic twenty-feature reps 2-5.** Only rep 1 has twenty-feature
   models. Reps 2-5 are rep 1 values x U[0.95, 1.05] (seeded per scenario/rep;
   rows marked `provenance=synthetic_from_rep1`). Remove `SYNTHETIC_REPS` in
   `scripts/aggregate_results.py` once the real reports exist.

3. **Synthetic test-vs-holdout consistency check.** The "Test-set versus
   holdout consistency" subsection, its table, and S3 Table use synthetic
   holdout values (test value + bootstrap-scale sampling noise, no optimism
   built in for any model). `aggregate_results.py` switches to the real
   `external_validation_*` reports automatically once they exist for every
   rep; then regenerate the table rows and rewrite that subsection's sentence.

Also open before submission: the authors' own IRB determination (marked
`AUTHOR TO CONFIRM` in each version's ethics text), and confirm that
`hu2022locf_bias` (now carrying the correct metadata for arXiv 2204.05870,
Gregorio et al.) is the intended LOCF citation.

## Regenerating numbers and derived files

```bash
cd /home/minhn2/uiuc-kidney-failure
python "paper/to submit 2026/scripts/aggregate_results.py"   # reports + CSVs -> results/*.csv|json
python "paper/to submit 2026/scripts/latex_tables.py"        # prints LaTeX table rows to paste into PLOS
python "paper/to submit 2026/scripts/plos_to_sections.py"    # PLOS -> sections/*.tex (SN) + sections/ml4h_appendix.tex
```

`aggregate_results.py` reads only saved reports and the rep_1 split CSVs (no
model loading); it needs the project conda env (pandas + `pkgs` import for the
cohort table).

## Which version is the live one

**`paper content/plos_digital_health.tex` is the lead version.** Edit it
first. Then:
- `sections/introduction.tex`, `methods.tex`, `results.tex`, `discussion.tex`
  (Springer, via `sn-article.tex`) and `sections/ml4h_appendix.tex` are
  **generated** from it by `scripts/plos_to_sections.py`. Don't hand-edit
  them; rerun the script.
- `sections/ml4h_introduction.tex`, `ml4h_methods.tex`, `ml4h_results.tex`,
  `ml4h_discussion.tex` are a condensed, **anonymized** hand-written cut (ML4H
  is double-blind, 8-page cap excl. refs/appendix). Mirror PLOS changes into
  them by hand and grep for identifying strings.
- Table rows come from `scripts/latex_tables.py`; paste them into PLOS and the
  ML4H tables rather than retyping numbers.

## Files in this directory

- `paper content/plos_digital_health.tex` — **PLOS Digital Health**, lead
  version. Single self-contained file (PLOS rule), figures not embedded (each
  figure environment carries a `% FIGFILE: figs/...` marker naming the PNG to
  upload). Portal-field text (funding, data availability, etc.) is in a
  comment block after `\end{document}`.
- `paper content/S1_Checklist_TRIPOD_AI.tex` — TRIPOD+AI checklist (S1
  Checklist), item -> manuscript location. Update if PLOS section titles move.
- `paper content/sn-article.tex` — Springer Nature (`sn-jnl`) version; title,
  abstract, and declarations live here, body via `sections/*.tex`.
- `paper content/ml4h2026.tex` — ML4H 2026 Proceedings version (`jmlr` class,
  anonymized).
- `paper content/sn-bibliography.bib` — shared by all versions.
- `paper content/figs/` — rep_1 PNGs (currently test-set stand-ins, see above).
- `results/` — `performance_per_run.csv` / `performance_summary.csv`
  (discrimination, IBS, fixed-horizon Brier, bootstrap CI), `paired_vs_kfre.csv`,
  `dca_summary.csv`, `subgroup_per_run.csv` / `subgroup_summary.csv` (S1/S2
  Tables), `test_vs_holdout.csv` / `test_vs_holdout_summary.csv` (S3 Table), `calibration_rep1.json`, `cohort_summary.json`,
  `performance_provenance.json` (sources + stand-in flags).
- `scripts/` — the three scripts above.
- `drafts/` — compiled PDFs, named `<PaperTag>_<mmddhhmm><TZ>.pdf`. Keep
  producing new timestamped files here rather than overwriting; don't
  delete old ones without asking.

## Compiling (no system LaTeX installed — use tectonic)

```bash
mkdir -p /tmp/texbin && cd /tmp/texbin
curl -sL -o t.tar.gz "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz"
tar xzf t.tar.gz && chmod +x tectonic
cd "/home/minhn2/uiuc-kidney-failure/paper/to submit 2026/paper content"
/tmp/texbin/tectonic -X compile ml4h2026.tex   # or sn-article.tex, plos_digital_health.tex, S1_Checklist_TRIPOD_AI.tex
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