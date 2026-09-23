"""Audit of prediction time, input cutoff and outcome horizon
(PAPER_GAPS_EXPERIMENT_PLAN.md Gap 8).

How to run
----------
From the repository root, with the project Python environment active::

    # Static + exported-data audit for rep99 (seconds; reads only generated_data/):
    CKD_REP=99 python -m pkgs.scripts.audit_prediction_time

    # Selected scenarios:
    CKD_REP=1 python -m pkgs.scripts.audit_prediction_time --scenarios four_features

    # Additionally quantify matched-lab timing from raw MIMIC labevents
    # (SLOW -- a chunked scan of the 13.7 GB labevents.csv; run it in the
    # background and expect tens of minutes):
    CKD_REP=99 python -m pkgs.scripts.audit_prediction_time --lab-timing

Writes generated_data/rep<N>/stage_gap8_prediction_time_audit_report.txt.

What this audits, and what it deliberately does not do
------------------------------------------------------
Gap 8 is explicit that the extraction's matching rules are DOCUMENTED, ACCEPTED
tradeoffs, not unexplained mistakes, and that the next step is to "document the
prediction target and check temporal alignment for each model" and "quantify
later matches and time separations to assess the limitation" -- and equally
explicit that it does NOT prescribe backward matching, duration resets or a
re-extraction before that audit. So this script changes no extraction rule and
regenerates no cohort. It reports three things:

Part A -- the prediction target of each of the 11 models, read off the code path
that actually produces its predictions (which landmark row it is scored on,
which inputs that row carries, what the outcome is, and over what horizon), not
off a comment describing it.

Part B -- temporal facts recoverable from the exported scenario CSVs alone:
the time origin, where each patient's landmark sits, how much follow-up exists
past the reported 2-year and 5-year horizons, and how much of the cohort's
"risk" is actually observable inside those windows.

Part C (--lab-timing) -- the one question the exported CSVs cannot answer:
how often a matched lab was recorded AFTER the creatinine draw the row is
anchored on, and by how much. The exported frames keep no absolute timestamp,
so this re-derives the match under CURRENT rules from raw labevents for the rep's own cohort,
calling the extraction's own merge_nearest_within_admission() rather than
re-implementing it. This is not a reconstruction of historical exports if their
matching rules differed; uACR now uses backward matching.

A finding here is a description of what the reconstruction did. It does not by
itself establish that any model's score is or is not usable as prospective
risk -- that conclusion belongs in the manuscript's framing, which Gap 8 and
Gap 6 coordinate on.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCENARIOS = ('four_features', 'eight_features', 'twenty_features_heterogeneous')
HORIZONS_DAYS = (730, 1825)

# Part A. One row per model, describing the prediction target its OWN
# predictions() method implements (pkgs/models/*.py), verified against that code
# rather than against any comment about it.
MODEL_PREDICTION_TARGETS = [
    # (model, landmark the score is attached to, inputs consumed, native horizon support)
    ('Cox (CoxTimeVaryingFitter)',
     "each patient's LAST exported row (get_last_observation_data)",
     "that row's covariates only; predict_partial_hazard needs no time interval",
     "baseline_cumulative_hazard_ from the fit, read at any t it covers; "
     "covariates held at the landmark value thereafter"),
    ('Dynamic-DeepHit',
     "the patient (one curve per subject; the whole sequence is pooled by attention)",
     "every exported row of the patient, up to the last one",
     "per-day PMF over 0-5474 d; CIF read directly at any day in range"),
    ('Hazard Transformer',
     "the patient (one curve per subject; whole sequence pooled)",
     "every exported row of the patient, up to the last one",
     "100 bins over 0-730 d; no native prediction past 730 d"),
    ('Logistic Hazard',
     "each patient's LAST exported row (LogisticHazardDataset takes iloc[-1])",
     "that row's covariates only",
     "50 quantile bins over the training duration range, via LabTransDiscreteTime"),
    ('RNN-Surv',
     "each patient's LAST exported row, as a length-1 sequence",
     "that row's covariates only",
     "PMF bins over 0-730 d"),
    ('DeepSurv',
     "each patient's LAST exported row",
     "that row's covariates only",
     "none (score only); survival probabilities come from the labelled fitted conversion"),
    ('GBSA',
     "each patient's LAST exported row",
     "that row's covariates only",
     "predict_survival_function (loss='coxph' Breslow baseline), to the training max time"),
    ('Survival RF',
     "each patient's LAST exported row",
     "that row's covariates only",
     "predict_survival_function (ensemble Nelson-Aalen), to the training max time"),
    ('Survival SVM',
     "each patient's LAST exported row",
     "that row's covariates only",
     "none (pure ranking model); survival probabilities come from the labelled fitted conversion"),
    ('Weibull AFT',
     "each patient's LAST exported row",
     "that row's covariates only",
     "closed-form parametric S(t|x), defined at every t > 0"),
    ('KFRE',
     "each patient's LAST exported row",
     "that row's age/sex/eGFR/uACR (+ albumin/phosphate/bicarbonate/calcium for 8-var)",
     "exactly 2 y and 5 y -- the only horizons the published S0 constants define"),
]


def _report(lines, message):
    print(message, flush=True)
    lines.append(message)


def audit_static(lines):
    _report(lines, "=" * 100)
    _report(lines, "PART A — PREDICTION TARGET PER MODEL (read from pkgs/models/*.py's predictions())")
    _report(lines, "=" * 100)
    _report(lines, "")
    _report(lines, "Shared across all 11 models:")
    _report(lines, "  Time origin   : days since that patient's FIRST lab record in the exported frame")
    _report(lines, "                  (time_series_utils_store.calculate_duration_in_days). The origin is")
    _report(lines, "                  NOT reset at the landmark, so a reported 730-day horizon is measured")
    _report(lines, "                  from the first record, not from the row the score is attached to.")
    _report(lines, "  Outcome       : has_esrd, from ICD diagnosis codes (pkgs/commons.esrd_codes).")
    _report(lines, "                  ESRD-positive patients' rows are truncated at the first ESRD")
    _report(lines, "                  diagnosis DATE and only rows on that date carry has_esrd=1")
    _report(lines, "                  (time_series_store.process_patient_esrd); ESRD-negative patients")
    _report(lines, "                  are censored at their last lab record.")
    _report(lines, "  Landmark      : every model is scored on one row/curve per patient. Nine models")
    _report(lines, "                  read the patient's LAST exported row; Dynamic-DeepHit and Hazard")
    _report(lines, "                  Transformer pool the whole sequence up to that row.")
    _report(lines, "")
    _report(lines, "  CONSEQUENCE, stated plainly: for an ESRD-positive patient the landmark row is the")
    _report(lines, "  last row at or before the diagnosis date, so the input cutoff is the outcome date,")
    _report(lines, "  not a decision time some fixed interval before it. The duration attached to that")
    _report(lines, "  patient is time from first record to diagnosis, which is a follow-up length, not a")
    _report(lines, "  prediction lead time. Part B quantifies how large that distinction is.")
    _report(lines, "")
    header = f"{'Model':<28} {'Landmark':<58} {'Inputs':<62} Native horizon support"
    _report(lines, header)
    _report(lines, "-" * len(header))
    for model, landmark, inputs, horizon in MODEL_PREDICTION_TARGETS:
        _report(lines, f"{model:<28} {landmark:<58} {inputs:<62} {horizon}")
    _report(lines, "")


def audit_exported(lines, scenario_name, df_train, df_test):
    from pkgs.data_analysis.patient_outcomes import patient_level_outcomes

    _report(lines, "=" * 100)
    _report(lines, f"PART B — TEMPORAL ALIGNMENT FROM THE EXPORTED DATA — {scenario_name}")
    _report(lines, "=" * 100)
    for split_name, df in (('train', df_train), ('test', df_test)):
        terminal = patient_level_outcomes(df)
        durations = terminal['duration_in_days'].values.astype(float)
        events = terminal['has_esrd'].values.astype(bool)
        rows_per_patient = df.groupby('subject_id').size()

        _report(lines, "")
        _report(lines, f"[{split_name}] {len(df)} lab-event rows over {len(terminal)} patients "
                       f"({rows_per_patient.median():.0f} rows/patient median, "
                       f"{rows_per_patient.max()} max)")
        _report(lines, f"  Landmark (days from first record to last): median {np.median(durations):.1f}, "
                       f"p90 {np.percentile(durations, 90):.1f}, max {durations.max():.1f}")
        _report(lines, f"  Events: {int(events.sum())}/{len(events)} "
                       f"({events.mean() * 100:.1f}%)")
        if events.any():
            _report(lines, f"  Time from first record to EVENT (ESRD-positive patients): "
                           f"median {np.median(durations[events]):.1f} d, "
                           f"p10 {np.percentile(durations[events], 10):.1f} d, "
                           f"min {durations[events].min():.1f} d")
        if (~events).any():
            _report(lines, f"  Time from first record to CENSORING (ESRD-negative patients): "
                           f"median {np.median(durations[~events]):.1f} d, "
                           f"max {durations[~events].max():.1f} d")

        n_zero = int((durations <= 0).sum())
        n_single = int((rows_per_patient == 1).sum())
        _report(lines, f"  Patients with zero follow-up duration: {n_zero}/{len(durations)} "
                       f"({n_zero / len(durations) * 100:.1f}%); these can have multiple rows.")
        _report(lines, f"  Patients with a single exported row: {n_single}/{len(durations)} "
                       f"({n_single / len(durations) * 100:.1f}%).")

        for horizon in HORIZONS_DAYS:
            at_risk_past = int((durations >= horizon).sum())
            events_within = int((events & (durations <= horizon)).sum())
            censored_within = int(((~events) & (durations < horizon)).sum())
            _report(lines, f"  Horizon {horizon} d: {events_within} events observed inside it; "
                           f"{censored_within} patients censored before it (outcome unobserved); "
                           f"{at_risk_past} patients followed to or past it.")
        _report(lines, "  NOTE: a patient censored before a horizon contributes no outcome information "
                       "at that horizon; the IPCW-weighted metrics reweight for this, the raw "
                       "proportions above do not.")
    _report(lines, "")


_LABS_CACHE = {}


def _load_cohort_labs(lines, subject_ids, chunksize=2_000_000):
    """Creatinine/uACR/chemistry rows for these patients, read once from
    labevents.csv in chunks and cached for the rest of the process.

    labevents.csv is 13.7 GB / ~118M rows and the extraction's own loader reads
    all of it into memory; this restricts the read to five columns and drops
    everything outside the cohort and the item ids of interest chunk by chunk,
    which keeps the result in the low hundreds of thousands of rows. The cache
    is keyed on the patient set so running several scenarios in one invocation
    scans the file once instead of once per scenario."""
    from pkgs.commons import (lab_events_file_path, lab_codes_creatinine, lab_codes_uacr,
                              lab_codes_calcium, lab_codes_phosphate, lab_codes_bicarbonate,
                              lab_codes_serum_albumin)
    wanted = {
        'creatinine': set(lab_codes_creatinine), 'uacr': set(lab_codes_uacr),
        'calcium': set(lab_codes_calcium), 'phosphate': set(lab_codes_phosphate),
        'bicarbonate': set(lab_codes_bicarbonate), 'serum_albumin': set(lab_codes_serum_albumin),
    }
    key = frozenset(subject_ids)
    if key in _LABS_CACHE:
        return _LABS_CACHE[key], wanted
    for cached_ids, cached_labs in _LABS_CACHE.items():
        if key.issubset(cached_ids):
            return cached_labs[cached_labs['subject_id'].isin(key)].copy(), wanted

    all_codes = set().union(*wanted.values())
    _report(lines, f"  Scanning {lab_events_file_path} for {len(subject_ids)} cohort patients "
                   f"and {len(all_codes)} lab item ids (chunked, {chunksize:,} rows at a time)...")
    kept, n_scanned = [], 0
    for chunk in pd.read_csv(lab_events_file_path, chunksize=chunksize,
                             usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum']):
        n_scanned += len(chunk)
        chunk = chunk[chunk['subject_id'].isin(subject_ids)]
        if len(chunk) == 0:
            continue
        chunk['itemid'] = chunk['itemid'].astype(str)
        chunk = chunk[chunk['itemid'].isin(all_codes)]
        if len(chunk):
            kept.append(chunk)
    labs = pd.concat(kept) if kept else pd.DataFrame(
        columns=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'])
    if len(labs):
        labs['charttime'] = pd.to_datetime(labs['charttime'])
    _report(lines, f"  Scanned {n_scanned:,} labevents rows; kept {len(labs):,} for this cohort.")
    _LABS_CACHE[key] = labs
    return labs, wanted


def audit_lab_timing(lines, scenario_name, df_test, df_train):
    """Part C — how often a matched lab was recorded AFTER the creatinine draw
    its row is anchored on, and by how much.

    Re-derives the match for THIS REP'S OWN COHORT and calls the extraction's
    own merge_nearest_within_admission() so the rule audited is the one the
    pipeline actually uses. The matched row's timestamp comes back from that
    merge directly (the `<lab>_charttime` column it now carries), so the
    separation is exact rather than reconstructed. It does not rebuild the
    cohort or rewrite any exported file."""
    from pkgs.data_analysis.time_series_store import merge_nearest_within_admission

    _report(lines, "=" * 100)
    _report(lines, f"PART C — MATCHED-LAB TIMING RELATIVE TO THE ANCHOR CREATININE DRAW — {scenario_name}")
    _report(lines, "=" * 100)
    if scenario_name not in ('four_features', 'eight_features'):
        _report(lines, f"  Not applicable: {scenario_name} anchors every row on its own single drawn lab "
                       "and performs no nearest-time matching, so there is no anchor/matched pair to time.")
        _report(lines, "")
        return

    subject_ids = set(df_train['subject_id'].unique()) | set(df_test['subject_id'].unique())
    labs, wanted = _load_cohort_labs(lines, subject_ids)
    if len(labs) == 0:
        _report(lines, "  No rows kept — nothing to time.")
        _report(lines, "")
        return

    anchor = labs[labs['itemid'].isin(wanted['creatinine'])].dropna(subset=['hadm_id']).copy()
    anchor = anchor[anchor['valuenum'].notna() & (anchor['valuenum'] != 0)]
    _report(lines, f"  Anchor rows (creatinine draws with an admission, non-zero value): {len(anchor):,} "
                   f"over {anchor['subject_id'].nunique()} patients.")
    _report(lines, "  Rule audited per lab, exactly as the extraction applies it "
                   "(time_series_store.merge_nearest_within_admission):")
    _report(lines, "  Recomputed with current matching rules on raw anchors for cohort patients; "
                   "this does not validate previously exported feature values.")

    matched_labs = [('uacr', 'subject_id', None, 'backward')]
    if scenario_name == 'eight_features':
        matched_labs += [(name, 'hadm_id', pd.Timedelta(hours=24), 'nearest')
                         for name in ('calcium', 'phosphate', 'bicarbonate', 'serum_albumin')]

    for lab_name, by, tolerance, direction in matched_labs:
        other = labs[labs['itemid'].isin(wanted[lab_name])].copy()
        other[lab_name] = other['valuenum']
        other = other.dropna(subset=[by, 'charttime', lab_name])
        if len(other) == 0:
            _report(lines, f"    {lab_name:<15} no source rows in this cohort — skipped.")
            continue
        merged = merge_nearest_within_admission(anchor.copy(), other, lab_name,
                                                tolerance=tolerance, by=by, direction=direction)
        time_col = f'{lab_name}_charttime'
        matched = merged[merged[lab_name].notna() & merged[time_col].notna()]
        deltas = (matched[time_col] - matched['charttime']).dt.total_seconds() / 3600.0
        if len(deltas) == 0:
            _report(lines, f"    {lab_name:<15} no matches with a recoverable timestamp.")
            continue
        after = int((deltas > 0).sum())
        simultaneous = int((deltas == 0).sum())
        window = 'same admission, +/-24 h' if tolerance is not None else (
            "patient's history at/before anchor, no maximum lookback" if by == 'subject_id' else 'same admission')
        _report(lines, f"    {lab_name:<15} matched by {by} ({window}; direction={direction})")
        _report(lines, f"      matched {len(matched):,}/{len(merged):,} anchor rows "
                       f"({len(matched) / len(merged) * 100:.1f}%)")
        _report(lines, f"      recorded AFTER the anchor creatinine: {after:,}/{len(deltas):,} "
                       f"({after / len(deltas) * 100:.1f}%); simultaneous: {simultaneous:,} "
                       f"({simultaneous / len(deltas) * 100:.1f}%)")
        _report(lines, f"      signed separation (matched - anchor), hours: "
                       f"median {deltas.median():+.2f}, "
                       f"p5 {deltas.quantile(0.05):+.2f}, p95 {deltas.quantile(0.95):+.2f}, "
                       f"max after {deltas.max():+.2f}, max before {deltas.min():+.2f}")
        _report(lines, f"      absolute separation, hours: median {deltas.abs().median():.2f}, "
                       f"p95 {deltas.abs().quantile(0.95):.2f}, max {deltas.abs().max():.2f}")
    _report(lines, "")
    _report(lines, "  Reading: a positive separation means the covariate was measured after the")
    _report(lines, "  creatinine the row is anchored on, so that row's feature vector contains")
    _report(lines, "  information not available at the anchor timestamp. Whether that matters depends")
    _report(lines, "  on the claim being made: for a retrospective benchmark on a reconstructed")
    _report(lines, "  snapshot it is a documented property of the reconstruction; for a prospective")
    _report(lines, "  claim it is an information cutoff that would have to be defined and enforced.")
    _report(lines, "  Quantifying it does not settle which claim the manuscript makes — see Gap 8.")
    _report(lines, "")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument('--lab-timing', action='store_true',
                        help='Also run Part C, which scans the raw labevents.csv (slow)')
    args = parser.parse_args(argv)

    from pkgs.commons import current_rep, generate_data_path_latest_rep
    from pkgs.data_analysis.model_data_store import get_train_test_data
    from pkgs.data_analysis.types import ExperimentScenario

    output_dir = Path(generate_data_path_latest_rep)
    lines = []
    _report(lines, f"Repetition: {current_rep}   Generated: "
                   f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _report(lines, "")
    audit_static(lines)

    frames = {}
    for scenario_name in args.scenarios:
        for split in ('train', 'test'):
            path = output_dir / f'{scenario_name}_{split}_data.csv'
            if not path.is_file():
                _report(lines, f"SKIP {scenario_name}: missing {path}")
                break
        else:
            frames[scenario_name] = get_train_test_data(ExperimentScenario(scenario_name))

    # Scan raw labs once for the union, then restrict cached rows per scenario.
    if args.lab_timing:
        subject_ids = set()
        for scenario_name, pair in frames.items():
            if scenario_name in ('four_features', 'eight_features'):
                for df in pair:
                    subject_ids.update(df['subject_id'].unique())
        if subject_ids:
            _load_cohort_labs(lines, subject_ids)

    for scenario_name, (df_train, df_test) in frames.items():
        audit_exported(lines, scenario_name, df_train, df_test)
        if args.lab_timing:
            audit_lab_timing(lines, scenario_name, df_test, df_train)

    report_path = output_dir / 'stage_gap8_prediction_time_audit_report.txt'
    header = [
        "GAP 8 — PREDICTION TIME / INPUT CUTOFF / OUTCOME HORIZON AUDIT",
        "=" * 100,
        "See PAPER_GAPS_EXPERIMENT_PLAN.md Gap 8. This audit changes no extraction rule and",
        "regenerates no cohort; it describes what the existing reconstruction does.",
        "=" * 100,
        "",
    ]
    report_path.write_text('\n'.join(header + lines))
    print(f"Audit report saved to: {report_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
