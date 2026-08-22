"""Cohort-flow analysis for four_features / eight_features / twenty_features_heterogeneous.

Produces a STROBE/TRIPOD-style participant-flow report per scenario: final N patients/records
compared against the original (pre-lab-filtering) source cohort, plus outcome (has_esrd) rate at
both. See EXPERIMENT_PLAN_DETAILS.md, section "1c-0", for the reporting-guideline citations and
rationale this follows:
  - STROBE (von Elm E, Altman DG, Egger M, et al. "The Strengthening the Reporting of Observational
    Studies in Epidemiology (STROBE) Statement." Lancet. 2007;370(9596):1453-1457), item 13(a).
  - TRIPOD (Collins GS, Reitsma JB, Altman DG, Moons KGM. "Transparent Reporting of a Multivariable
    Prediction Model for Individual Prognosis or Diagnosis (TRIPOD): The TRIPOD Statement." Ann
    Intern Med. 2015;162(1):55-63), item 13a.

Note: time_series_store.py's merge branches also emit intermediate "COHORT_FLOW|..." lines to the
extraction log at each merge stage (source population -> has eGFR -> has qualifying uACR -> ...).
That's left in place as useful debugging instrumentation for the merge itself, but this report
deliberately does not surface it - it's raw-merge-stage detail, not something worth reporting
alongside the actual result (per user direction 2026-08-21); it also measures a different, earlier
point in the pipeline than the final has_esrd-labeled cohort, so it doesn't predict final N anyway.
"""
import time

import pandas as pd

from pkgs.commons import (
    four_features_train_data_path, four_features_test_data_path,
    eight_features_train_data_path, eight_features_test_data_path,
    twenty_features_heterogeneous_train_data_path, twenty_features_heterogeneous_test_data_path,
    generate_data_path_latest_rep, current_rep,
    diagnose_icd_file_path, ckd_codes_stage3_to_5, esrd_codes,
)
from pkgs.data_analysis.types import ExperimentScenario

TRAIN_TEST_PATHS = {
    ExperimentScenario.FOUR_FEATURES: (four_features_train_data_path, four_features_test_data_path),
    ExperimentScenario.EIGHT_FEATURES: (eight_features_train_data_path, eight_features_test_data_path),
    ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS: (
        twenty_features_heterogeneous_train_data_path, twenty_features_heterogeneous_test_data_path),
}


def get_original_cohort_counts():
    """The pre-lab-filtering source cohort every scenario in this repo (old and new) is actually
    built from, per get_time_series_data_ckd_patients: patients with a CKD stage 3-5 diagnosis
    code OR an ESRD diagnosis code (NOT get_ckd_patients_and_diagnoses(late_stage=True), which
    only filters on CKD-3-5 codes and misses patients who went straight to an ESRD code without a
    separately-logged CKD-3-5 stage - that undercounts the true source population, as found during
    the 1c-0 pilot: 10,179 vs the correct 34,332). Cheap - only reads diagnoses_icd.csv, no
    labevents.csv. Returns (n_patients, n_records, n_esrd_positive_patients)."""
    diagnoses_df = pd.read_csv(diagnose_icd_file_path)
    diagnoses_df = diagnoses_df[diagnoses_df['icd_code'].isin(ckd_codes_stage3_to_5 + esrd_codes)]
    esrd_patients = diagnoses_df[diagnoses_df['icd_code'].isin(esrd_codes)]['subject_id'].unique()
    return diagnoses_df['subject_id'].nunique(), len(diagnoses_df), len(esrd_patients)


def analyze_cohort_flow(scenario: ExperimentScenario, out_path: str = None):
    assert scenario in TRAIN_TEST_PATHS, f"No cohort-flow analysis defined for {scenario}"

    train_path, test_path = TRAIN_TEST_PATHS[scenario]
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    orig_patients, orig_records, orig_esrd_patients = get_original_cohort_counts()
    orig_positive_rate = 100 * orig_esrd_patients / orig_patients if orig_patients else 0.0

    combined = pd.concat([train_df, test_df], ignore_index=True)
    final_patients = combined['subject_id'].nunique()
    final_records = len(combined)
    # Patient-level: a patient counts as positive if ANY of their rows has has_esrd=1 (the
    # terminal row, per process_positive_patients). This is the TRIPOD-comparable "participants
    # with the outcome" count - not the same as the row-level rate (most rows for a positive
    # patient are still has_esrd=0; only their last row is flagged).
    final_positive_patients = combined.loc[combined['has_esrd'] == 1, 'subject_id'].nunique()
    final_positive_patient_rate = 100 * final_positive_patients / final_patients if final_patients else 0.0
    final_positive_records = int(combined['has_esrd'].sum())
    final_positive_record_rate = 100 * final_positive_records / final_records if final_records else 0.0

    lines = []
    lines.append(f"COHORT FLOW ANALYSIS - {scenario.value}")
    lines.append("=" * 80)
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  (rep{current_rep})")
    lines.append(
        "Reporting basis: STROBE (von Elm et al. 2007, Lancet, 370(9596):1453-1457) item 13(a); "
        "TRIPOD (Collins et al. 2015, Ann Intern Med, 162(1):55-63) item 13a. See "
        "EXPERIMENT_PLAN_DETAILS.md \"1c-0\" for the full citations and rationale.")
    lines.append("")

    lines.append("MINIMUM (required)")
    lines.append("-" * 80)
    lines.append("Original source cohort (pre-lab-filtering: CKD stage 3-5 diagnosis code OR ESRD")
    lines.append("diagnosis code - same base population get_time_series_data_ckd_patients builds")
    lines.append("every scenario in this repo, old and new, from):")
    lines.append(f"  N patients:          {orig_patients:,}")
    lines.append(f"  N diagnosis records: {orig_records:,}")
    lines.append(f"  Outcome-positive (ESRD dx) rate: {orig_esrd_patients:,} / {orig_patients:,} ({orig_positive_rate:.2f}%)")
    lines.append("")
    lines.append(f"Final extracted cohort for {scenario.value} (train+test combined):")
    lines.append(f"  N patients: {final_patients:,}")
    lines.append(f"  N records:  {final_records:,}")
    lines.append(
        f"  Attrition:  {100 * final_patients / orig_patients:.2f}% of original patients retained "
        f"({orig_patients:,} -> {final_patients:,})")
    lines.append("")

    lines.append("RECOMMENDED (flagged for approval, not separately mandated - see EXPERIMENT_PLAN_DETAILS.md)")
    lines.append("-" * 80)
    lines.append("Outcome (has_esrd) rate, patient-level (TRIPOD 13a: \"number of participants with")
    lines.append("and without the outcome\" - a patient counts positive if their terminal row is")
    lines.append("flagged has_esrd=1; not the same as the row-level rate below, since only one row")
    lines.append("per ESRD-positive patient is actually flagged):")
    lines.append(f"  Source cohort:        {orig_esrd_patients:,} / {orig_patients:,} ({orig_positive_rate:.2f}%) positive")
    lines.append(f"  Final extracted ({scenario.value}): {final_positive_patients:,} / {final_patients:,} ({final_positive_patient_rate:.2f}%) positive")
    lines.append(
        f"  Shift: {final_positive_patient_rate - orig_positive_rate:+.2f} percentage points "
        "- per TRIPOD 13a, this checks whether the merge-window filter (1a-2) disproportionately")
    lines.append("  drops ESRD-positive or -negative patients. A shift near 0 means the filter is close to outcome-neutral.")
    lines.append("")
    lines.append(f"Outcome (has_esrd) rate, row-level (fraction of individual records flagged positive,")
    lines.append(f"not directly comparable to the source cohort's patient-level rate above):")
    lines.append(f"  {final_positive_records:,} / {final_records:,} ({final_positive_record_rate:.2f}%) positive")
    lines.append("")

    lines.append("Train/test split:")
    lines.append(
        f"  Train: {train_df['subject_id'].nunique():,} patients, {len(train_df):,} records, "
        f"{100 * train_df['has_esrd'].sum() / len(train_df):.2f}% positive")
    lines.append(
        f"  Test:  {test_df['subject_id'].nunique():,} patients, {len(test_df):,} records, "
        f"{100 * test_df['has_esrd'].sum() / len(test_df):.2f}% positive")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    if out_path is None:
        out_path = f'{generate_data_path_latest_rep}/{scenario.value}_cohort_flow_report.txt'
    with open(out_path, 'w') as f:
        f.write(report + "\n")
    print(f"\nReport written to {out_path}")
    return out_path


if __name__ == '__main__':
    for scenario in [ExperimentScenario.FOUR_FEATURES, ExperimentScenario.EIGHT_FEATURES, ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS]:
        analyze_cohort_flow(scenario)
