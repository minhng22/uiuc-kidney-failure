"""Audit of the labelled outcome and the cohort eligibility rule, for the KFRE
comparison (PAPER_GAPS_EXPERIMENT_PLAN.md Gap 6).

How to run
----------
From the repository root, with the project Python environment active::

    CKD_REP=99 python -m pkgs.scripts.audit_outcome_definition

Reads diagnoses_icd.csv, patients.csv and admissions.csv (all small; it does NOT
touch labevents.csv) plus the exported scenario CSVs for whichever rep CKD_REP
selects. Writes generated_data/rep<N>/stage_gap6_outcome_definition_report.txt.

What this is for
----------------
Gap 6 is a reporting/interpretation gap, not a claim that the benchmark is
invalid. The manuscript's Limitations already acknowledge that a selected
hospital cohort differs from KFRE's validation populations. What is missing is a
verified statement of HOW the two differ, specifically in:

  (1) baseline eligibility — who is in the cohort at all,
  (2) outcome definition — what event the label marks, and when,
  (3) prediction horizon — over what window that event is predicted.

This script verifies (1) and (2) against the code and the data rather than
against prose, and reports the counts that let (3) be stated honestly. Gap 8's
audit (pkgs/scripts/audit_prediction_time.py) covers the timing side; the two
are meant to be read together.

Gap 6 also asks for one earlier characterisation to be RETRACTED: the claim that
the cohort's incidence is "one to two orders of magnitude lower" than KFRE's
populations. That comparison was never well formed — this benchmark's 83-92%
ESRD-positive proportion is a property of a deliberately balanced case-control
style extraction, not a fixed-horizon risk or an incidence rate, so it is not
commensurable with either. The report states that directly.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ICD classifications quoted below are from the published ICD-9-CM and ICD-10-CM
# tabular lists (WHO/CMS), not from a local dictionary file: MIMIC-IV's
# d_icd_diagnoses.csv is not present in this checkout's data/mimic-iv-2.2/hosp/
# (only admissions, d_labitems, diagnoses_icd, labevents, patients and
# prescriptions are), so the descriptions cannot be read back from the shipped
# data. Anything asserted here is checkable against the published tabular list.
ICD_MEANINGS = {
    'N17': 'ICD-10  Acute kidney failure, unspecified/with lesion (N17.x block)',
    'N170': 'ICD-10  Acute kidney failure with tubular necrosis',
    'N171': 'ICD-10  Acute kidney failure with acute cortical necrosis',
    'N172': 'ICD-10  Acute kidney failure with medullary necrosis',
    'N178': 'ICD-10  Other acute kidney failure',
    'N179': 'ICD-10  Acute kidney failure, unspecified',
    'N19': 'ICD-10  Unspecified kidney failure',
    '5845': 'ICD-9   Acute kidney failure with lesion of tubular necrosis',
    '5846': 'ICD-9   Acute kidney failure with lesion of renal cortical necrosis',
    '5847': 'ICD-9   Acute kidney failure with lesion of renal medullary necrosis',
    '5848': 'ICD-9   Acute kidney failure with other specified pathological lesion',
    '5849': 'ICD-9   Acute kidney failure, unspecified',
    '5853': 'ICD-9   Chronic kidney disease, stage III (moderate)',
    '5854': 'ICD-9   Chronic kidney disease, stage IV (severe)',
    '5855': 'ICD-9   Chronic kidney disease, stage V',
    'N183': 'ICD-10  Chronic kidney disease, stage 3',
    'N184': 'ICD-10  Chronic kidney disease, stage 4',
    'N185': 'ICD-10  Chronic kidney disease, stage 5',
}
# The codes a reader would expect an "ESRD" label to be built from, and which
# this extraction does NOT use.
TRUE_ESRD_CODES = {
    'N186': 'ICD-10  End stage renal disease',
    '5856': 'ICD-9   Chronic kidney disease, stage V requiring chronic dialysis (ESRD)',
}
# Dialysis / transplant procedure-adjacent diagnosis codes, i.e. the kind of
# evidence KFRE's own endpoint (initiation of dialysis or preemptive transplant)
# is established from.
DIALYSIS_TRANSPLANT_CODES = {
    'Z992': 'ICD-10  Dependence on renal dialysis',
    'Z940': 'ICD-10  Kidney transplant status',
    'T861': 'ICD-10  Complications of kidney transplant (T86.1x block)',
    'V451': 'ICD-9   Renal dialysis status',
    'V420': 'ICD-9   Kidney replaced by transplant',
    '99673': 'ICD-9   Complications due to renal dialysis device/graft',
}


def _report(lines, message):
    print(message, flush=True)
    lines.append(message)


def audit_codes(lines, diagnoses):
    from pkgs.commons import esrd_codes, ckd_codes_stage3_to_5

    _report(lines, "=" * 100)
    _report(lines, "PART 1 — WHAT THE LABEL ACTUALLY MARKS")
    _report(lines, "=" * 100)
    _report(lines, "")
    _report(lines, "pkgs/commons.py's `esrd_codes` — the code set `has_esrd` is built from")
    _report(lines, "(store.get_first_time_esrd_df / time_series_store.process_patient_esrd):")
    _report(lines, "")
    counts = diagnoses['icd_code'].value_counts()
    patients_per_code = diagnoses.groupby('icd_code')['subject_id'].nunique()
    for code in esrd_codes:
        meaning = ICD_MEANINGS.get(code, '(not in this script\'s reference table)')
        _report(lines, f"  {code:<8} {meaning:<70} rows={int(counts.get(code, 0)):>8}  "
                       f"patients={int(patients_per_code.get(code, 0)):>7}")
    _report(lines, "")
    _report(lines, "  FINDING: every code in this set is an ACUTE or UNSPECIFIED kidney-failure code.")
    _report(lines, "  None of them is end-stage renal disease. The codes a reader would expect are:")
    for code, meaning in TRUE_ESRD_CODES.items():
        present = code in esrd_codes
        _report(lines, f"    {code:<8} {meaning:<70} in esrd_codes: {present}   "
                       f"rows in MIMIC={int(counts.get(code, 0)):>8}  "
                       f"patients={int(patients_per_code.get(code, 0)):>7}")
    _report(lines, "")
    _report(lines, "  So `has_esrd` marks a hospital admission carrying an acute/unspecified")
    _report(lines, "  kidney-failure diagnosis in a patient who also carries a CKD stage 3-5")
    _report(lines, "  diagnosis. It does not mark end-stage renal disease, and it does not mark")
    _report(lines, "  the KFRE endpoint (initiation of maintenance dialysis, or preemptive")
    _report(lines, "  transplantation). For reference, MIMIC-IV's own counts for the")
    _report(lines, "  dialysis/transplant-status codes that endpoint would be built from:")
    for code, meaning in DIALYSIS_TRANSPLANT_CODES.items():
        _report(lines, f"    {code:<8} {meaning:<70} rows={int(counts.get(code, 0)):>8}  "
                       f"patients={int(patients_per_code.get(code, 0)):>7}")
    _report(lines, "")
    _report(lines, "  EVENT TIME: store.get_first_time_esrd_df takes the ADMITTIME of the earliest")
    _report(lines, "  admission carrying one of those codes — an admission timestamp, not a")
    _report(lines, "  treatment-initiation date. time_series_store.process_patient_esrd then marks")
    _report(lines, "  every lab row on that calendar DATE as the event and discards the patient's")
    _report(lines, "  later rows, so the label's resolution is one day and the input cutoff")
    _report(lines, "  coincides with the event date (see Gap 8's audit).")
    _report(lines, "")
    _report(lines, "pkgs/commons.py's `ckd_codes_stage3_to_5` — the eligibility rule:")
    for code in ckd_codes_stage3_to_5:
        meaning = ICD_MEANINGS.get(code, '(not in this script\'s reference table)')
        _report(lines, f"  {code:<8} {meaning:<70} rows={int(counts.get(code, 0)):>8}  "
                       f"patients={int(patients_per_code.get(code, 0)):>7}")
    _report(lines, "  This one checks out: CKD stage 3-5 by diagnosis code, which is the same")
    _report(lines, "  disease stage band the original KFRE derivation cohorts were drawn from.")
    _report(lines, "")


def audit_cohort(lines, diagnoses):
    from pkgs.commons import esrd_codes, ckd_codes_stage3_to_5

    _report(lines, "=" * 100)
    _report(lines, "PART 2 — BASELINE ELIGIBILITY, AND WHY THE COHORT'S EVENT PROPORTION IS NOT AN INCIDENCE")
    _report(lines, "=" * 100)
    _report(lines, "")
    ckd_patients = set(diagnoses.loc[diagnoses['icd_code'].isin(ckd_codes_stage3_to_5), 'subject_id'])
    event_patients = set(diagnoses.loc[diagnoses['icd_code'].isin(esrd_codes), 'subject_id'])
    both = ckd_patients & event_patients
    all_patients = diagnoses['subject_id'].nunique()
    _report(lines, f"  MIMIC-IV patients with any diagnosis row          : {all_patients:>8}")
    _report(lines, f"  ... with a CKD stage 3-5 code (eligible)          : {len(ckd_patients):>8} "
                   f"({len(ckd_patients) / all_patients * 100:.2f}% of the above)")
    _report(lines, f"  ... with an `esrd_codes` code                     : {len(event_patients):>8}")
    _report(lines, f"  ... with BOTH (label-positive population)         : {len(both):>8} "
                   f"({len(both) / len(ckd_patients) * 100:.2f}% of eligible patients)")
    _report(lines, "")
    _report(lines, "  The extracted scenarios do NOT sample this population proportionally. Each rep")
    _report(lines, "  is built by drawing label-positive and label-negative patients separately")
    _report(lines, "  (time_series_store.process_esrd_patients / process_negative_patients, joined in")
    _report(lines, "  model_data_store), so the 83-92% label-positive proportion reported for the")
    _report(lines, "  extracted cohorts is a DESIGN PARAMETER of that draw plus the lab-availability")
    _report(lines, "  filtering that follows it — not an observed incidence and not a risk.")
    _report(lines, "")
    _report(lines, "  RETRACTION (Gap 6): the earlier claim that this cohort's incidence is 'one to")
    _report(lines, "  two orders of magnitude lower' than KFRE's validation populations should not be")
    _report(lines, "  repeated. It compared a sampled label proportion against a fixed-horizon risk,")
    _report(lines, "  which are not the same kind of quantity in either direction. The correct")
    _report(lines, "  statement is that the two are not commensurable, and why.")
    _report(lines, "")
    _report(lines, "  For the record, KFRE's own populations, so the difference can be described")
    _report(lines, "  rather than quantified against an incommensurable number:")
    _report(lines, "    - Tangri et al. 2011, JAMA 305(15):1553-1559 (PMID 21482743): two")
    _report(lines, "      nephrology-REFERRED CKD stage 3-5 cohorts in Ontario, development and")
    _report(lines, "      validation, with kidney-failure event proportions of 11% and 24%.")
    _report(lines, "    - Tangri et al. 2016, JAMA 315(2):164-174 (PMID 26757465): multinational")
    _report(lines, "      validation across 31 cohorts, extending beyond the referred setting.")
    _report(lines, "    - Endpoint in both: need for maintenance dialysis or preemptive kidney")
    _report(lines, "      transplantation. Horizons: 2 and 5 years.")
    _report(lines, "")


def audit_horizon(lines, scenario_names):
    from pkgs.commons import generate_data_path_latest_rep
    from pkgs.data_analysis.model_data_store import get_train_test_data
    from pkgs.data_analysis.patient_outcomes import patient_level_outcomes
    from pkgs.data_analysis.types import ExperimentScenario
    import numpy as np

    _report(lines, "=" * 100)
    _report(lines, "PART 3 — PREDICTION HORIZON SUPPORT IN THE EXTRACTED COHORTS")
    _report(lines, "=" * 100)
    _report(lines, "")
    _report(lines, "  KFRE is applied at its published 2-year and 5-year horizons without refitting,")
    _report(lines, "  while the learned models are fitted to this cohort. Whether those horizons are")
    _report(lines, "  supported here is a property of the extracted follow-up:")
    output_dir = Path(generate_data_path_latest_rep)
    for scenario_name in scenario_names:
        paths = [output_dir / f'{scenario_name}_{split}_data.csv' for split in ('train', 'test')]
        if not all(p.is_file() for p in paths):
            _report(lines, f"    {scenario_name}: exported data not present for this rep — skipped.")
            continue
        _, df_test = get_train_test_data(ExperimentScenario(scenario_name))
        terminal = patient_level_outcomes(df_test)
        durations = terminal['duration_in_days'].values.astype(float)
        events = terminal['has_esrd'].values.astype(bool)
        _report(lines, "")
        _report(lines, f"    {scenario_name}: {len(terminal)} held-out patients, "
                       f"{int(events.sum())} label-positive ({events.mean() * 100:.1f}%)")
        for horizon, label in ((730, '2-year'), (1825, '5-year')):
            observed = int((events & (durations <= horizon)).sum())
            censored_before = int(((~events) & (durations < horizon)).sum())
            followed = int((durations >= horizon).sum())
            _report(lines, f"      {label:<7} ({horizon} d): {observed} events inside the window, "
                           f"{censored_before} censored before it, {followed} followed to or past it")
        _report(lines, f"      follow-up: median {np.median(durations):.0f} d, "
                       f"p90 {np.percentile(durations, 90):.0f} d, max {durations.max():.0f} d")
    _report(lines, "")
    _report(lines, "  Reading: where most label-positive patients reach the event well inside two")
    _report(lines, "  years, a 2-year KFRE risk and a model score fitted on this cohort are being")
    _report(lines, "  asked different questions of different populations, and a head-to-head")
    _report(lines, "  discrimination difference between them reflects that as much as it reflects")
    _report(lines, "  model quality. Any superiority claim should be scoped to this cohort and this")
    _report(lines, "  protocol.")
    _report(lines, "")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scenarios', nargs='+',
                        choices=('four_features', 'eight_features', 'twenty_features_heterogeneous'),
                        default=['four_features', 'eight_features', 'twenty_features_heterogeneous'])
    args = parser.parse_args(argv)
    from pkgs.commons import current_rep, diagnose_icd_file_path, generate_data_path_latest_rep

    lines = []
    _report(lines, f"Repetition: {current_rep}   Generated: "
                   f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _report(lines, "")
    _report(lines, f"Reading {diagnose_icd_file_path} ...")
    diagnoses = pd.read_csv(diagnose_icd_file_path, usecols=['subject_id', 'hadm_id', 'icd_code'],
                            dtype={'icd_code': str})
    _report(lines, f"  {len(diagnoses):,} diagnosis rows, {diagnoses['subject_id'].nunique():,} patients.")
    _report(lines, "")

    audit_codes(lines, diagnoses)
    audit_cohort(lines, diagnoses)
    audit_horizon(lines, args.scenarios)

    _report(lines, "=" * 100)
    _report(lines, "PART 4 — WHAT THE MANUSCRIPT SHOULD SAY")
    _report(lines, "=" * 100)
    _report(lines, "")
    _report(lines, "  1. Name the outcome for what it is: an acute/unspecified kidney-failure")
    _report(lines, "     diagnosis code recorded in a patient with CKD stage 3-5, dated to the")
    _report(lines, "     admission that carried it. Do not call it ESRD without that qualification,")
    _report(lines, "     and do not describe it as the KFRE endpoint.")
    _report(lines, "  2. State that KFRE's endpoint is maintenance dialysis or preemptive")
    _report(lines, "     transplantation, that this differs from the labelled outcome here, and that")
    _report(lines, "     an equation applied to a different endpoint should not be expected to")
    _report(lines, "     calibrate — so a calibration gap is partly definitional, not only a")
    _report(lines, "     population effect.")
    _report(lines, "  3. Remove the 'one to two orders of magnitude lower' incidence comparison; say")
    _report(lines, "     instead that the extraction draws label-positive and label-negative patients")
    _report(lines, "     separately, so its label proportion is not an incidence and is not")
    _report(lines, "     comparable to KFRE's cohort risks.")
    _report(lines, "  4. State that the learned models are fitted to this cohort while KFRE is")
    _report(lines, "     applied without refitting, so the comparison is between a fitted and an")
    _report(lines, "     externally-specified model. Limit superiority claims to this cohort and")
    _report(lines, "     protocol.")
    _report(lines, "  5. Coordinate with Gap 8: the input cutoff coincides with the event date for")
    _report(lines, "     label-positive patients, which bears on whether any score here can be read")
    _report(lines, "     as prospective risk.")
    _report(lines, "")

    report_path = Path(generate_data_path_latest_rep) / 'stage_gap6_outcome_definition_report.txt'
    header = [
        "GAP 6 — OUTCOME DEFINITION, ELIGIBILITY AND HORIZON AUDIT FOR THE KFRE COMPARISON",
        "=" * 100,
        "See PAPER_GAPS_EXPERIMENT_PLAN.md Gap 6. This audit changes no code and no data; it",
        "verifies what the existing label and eligibility rule actually select.",
        "=" * 100,
        "",
    ]
    report_path.write_text('\n'.join(header + lines))
    print(f"Audit report saved to: {report_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
