"""Cohort-flow analysis for four_features / eight_features / twenty_features_heterogeneous.

Produces one "Table 1"-style baseline-characteristics table per scenario, source cohort vs. final
extracted cohort side by side, modeled directly on a real published external KFRE validation study's
own Table 1 rather than derived from a reporting checklist in the abstract:

  Major RW, Shepherd D, Medcalf JF, Xu G, Gray LJ, Brunskill NJ. "The Kidney Failure Risk Equation
  for prediction of end stage renal disease in UK primary care: An external validation and clinical
  impact projection cohort study." PLOS Medicine. 2019;16(11):e1002955.

See EXPERIMENT_PLAN_DETAILS.md, section "1c-0", for the per-row why+reference table this implements,
and for the citations (STROBE, TRIPOD, Tangri et al. 2016 eTable 1) backing individual choices (e.g.
why eGFR/uACR are reported both mean+SD and median+IQR, why there's no separate missing-data table).

Note: time_series_store.py's merge branches also emit intermediate "COHORT_FLOW|..." lines to the
extraction log at each merge stage (source population -> has eGFR -> has qualifying uACR -> ...).
That's left in place as useful debugging instrumentation for the merge itself, but this report
deliberately does not surface it - it's raw-merge-stage detail, not the actual result, and it
measures an earlier point in the pipeline than the final has_esrd-labeled cohort so it doesn't
predict final N anyway.
"""
import math
import time

import pandas as pd

from pkgs.commons import (
    four_features_train_data_path, four_features_test_data_path,
    eight_features_train_data_path, eight_features_test_data_path,
    twenty_features_heterogeneous_train_data_path, twenty_features_heterogeneous_test_data_path,
    generate_data_path_latest_rep, current_rep,
    diagnose_icd_file_path, ckd_codes_stage3_to_5, esrd_codes, patients_file_path,
)
from pkgs.data_analysis.types import ExperimentScenario

TRAIN_TEST_PATHS = {
    ExperimentScenario.FOUR_FEATURES: (four_features_train_data_path, four_features_test_data_path),
    ExperimentScenario.EIGHT_FEATURES: (eight_features_train_data_path, eight_features_test_data_path),
    ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS: (
        twenty_features_heterogeneous_train_data_path, twenty_features_heterogeneous_test_data_path),
}
DAYS_PER_YEAR = 365.25


def summarize(series):
    """Mean/SD/median/IQR for a numeric series, ignoring NaN. None if nothing to summarize."""
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) == 0:
        return None
    return {'n': len(s), 'mean': s.mean(), 'sd': s.std(), 'median': s.median(),
            'q1': s.quantile(0.25), 'q3': s.quantile(0.75)}


def fmt_mean_sd(stats, decimals=1):
    return "N/A" if stats is None else f"{stats['mean']:.{decimals}f} (SD {stats['sd']:.{decimals}f})"


def fmt_median_iqr(stats, decimals=1):
    if stats is None:
        return "N/A"
    return f"{stats['median']:.{decimals}f} (IQR {stats['q1']:.{decimals}f}–{stats['q3']:.{decimals}f})"


def incidence_rate_per_1000py(n_events, person_years):
    """Poisson incidence rate per 1,000 person-years with a 95% CI via the standard log-scale
    normal approximation (SE(log rate) = 1/sqrt(events)) - the same quantity Major et al. 2019
    Table 1 reports as "ESRD rate, per 1,000 person-years (95% CI)"."""
    if not n_events or person_years <= 0:
        return None
    rate = n_events / person_years * 1000
    se_log = 1 / math.sqrt(n_events)
    return {'rate': rate, 'lo': rate * math.exp(-1.96 * se_log), 'hi': rate * math.exp(1.96 * se_log)}


def fmt_rate(rate_stats):
    if rate_stats is None:
        return "N/A"
    return f"{rate_stats['rate']:.2f} (95% CI {rate_stats['lo']:.2f}–{rate_stats['hi']:.2f})"


def get_source_cohort_subject_ids_and_counts():
    """The pre-lab-filtering source cohort every scenario in this repo (old and new) is actually
    built from, per get_time_series_data_ckd_patients: patients with a CKD stage 3-5 diagnosis
    code OR an ESRD diagnosis code (NOT get_ckd_patients_and_diagnoses(late_stage=True), which
    only filters on CKD-3-5 codes and misses patients who went straight to an ESRD code without a
    separately-logged CKD-3-5 stage - that undercounts the true source population, as found during
    the 1c-0 pilot: 10,179 vs the correct 34,332). Cheap - only reads diagnoses_icd.csv, no
    labevents.csv."""
    diagnoses_df = pd.read_csv(diagnose_icd_file_path)
    diagnoses_df = diagnoses_df[diagnoses_df['icd_code'].isin(ckd_codes_stage3_to_5 + esrd_codes)]
    subject_ids = diagnoses_df['subject_id'].unique()
    esrd_patients = diagnoses_df[diagnoses_df['icd_code'].isin(esrd_codes)]['subject_id'].unique()
    return subject_ids, len(diagnoses_df), len(esrd_patients)


def get_age_gender_stats(subject_ids):
    """Cheap (patients.csv only, no labevents.csv) - works for any subject_id list, including the
    full source cohort, since age/gender live on patients.csv regardless of lab availability."""
    patients_df = pd.read_csv(patients_file_path)
    patients_df = patients_df[patients_df['subject_id'].isin(subject_ids)].drop_duplicates('subject_id')
    male_pct = 100 * (patients_df['gender'] == 'M').mean() if len(patients_df) else None
    return male_pct, summarize(patients_df['anchor_age'])


def build_final_cohort_column(scenario, combined):
    """Everything computable for free from the already-extracted train+test data - no additional
    labevents.csv reads."""
    col = {}
    col['n'] = combined['subject_id'].nunique()
    col['records'] = len(combined)

    male_pct, age_stats = get_age_gender_stats(combined['subject_id'].unique())
    col['male_pct'] = male_pct
    col['age'] = age_stats

    if scenario == ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS:
        col['egfr'] = summarize(combined.loc[combined['egfr_missing'] == 0, 'egfr'])
        col['uacr'] = None  # not one of the 20 features
        col['chem_panel'] = None  # eight_features-only row
    else:
        col['egfr'] = summarize(combined['egfr'])
        col['uacr'] = summarize(combined['uacr'])
        if scenario == ExperimentScenario.EIGHT_FEATURES:
            col['chem_panel'] = {lab: summarize(combined[lab])
                                  for lab in ['calcium', 'phosphate', 'bicarbonate', 'serum_albumin']}
        else:
            col['chem_panel'] = None

    per_patient_days = combined.groupby('subject_id')['duration_in_days'].max()
    per_patient_years = per_patient_days / DAYS_PER_YEAR
    positive_ids = combined.loc[combined['has_esrd'] == 1, 'subject_id'].unique()
    col['n_events'] = len(positive_ids)
    col['followup'] = summarize(per_patient_years)
    col['time_to_esrd'] = summarize(per_patient_years[per_patient_years.index.isin(positive_ids)])
    col['incidence_rate'] = incidence_rate_per_1000py(col['n_events'], per_patient_years.sum())
    return col


def build_source_cohort_column(subject_ids, n_records, n_events):
    """Source-cohort column. Only age/gender/n/events are computed here - eGFR, uACR, chem-panel,
    follow-up, and incidence rate for the *full* 34,332-patient source population would each
    require a fresh labevents.csv scan (the same order of cost as the original ~2h extraction) and
    are not computed for reporting purposes alone; available on request."""
    col = {'n': len(subject_ids), 'records': n_records, 'n_events': n_events}
    male_pct, age_stats = get_age_gender_stats(subject_ids)
    col['male_pct'] = male_pct
    col['age'] = age_stats
    col['egfr'] = col['uacr'] = col['chem_panel'] = None
    col['followup'] = col['time_to_esrd'] = col['incidence_rate'] = None
    return col


def analyze_cohort_flow(scenario: ExperimentScenario, out_path: str = None):
    assert scenario in TRAIN_TEST_PATHS, f"No cohort-flow analysis defined for {scenario}"

    train_path, test_path = TRAIN_TEST_PATHS[scenario]
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    combined = pd.concat([train_df, test_df], ignore_index=True)

    source_ids, source_records, source_events = get_source_cohort_subject_ids_and_counts()
    source = build_source_cohort_column(source_ids, source_records, source_events)
    final = build_final_cohort_column(scenario, combined)

    lines = []
    lines.append(f"COHORT FLOW ANALYSIS - {scenario.value}")
    lines.append("=" * 80)
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  (rep{current_rep})")
    lines.append(
        "Table modeled on Major et al. 2019, PLOS Medicine 16(11):e1002955, Table 1. See "
        "EXPERIMENT_PLAN_DETAILS.md \"1c-0\" for the per-row rationale and full citations.")
    lines.append("")

    def row(label, source_val, final_val):
        lines.append(f"  {label:<45} {source_val:<28} {final_val}")

    lines.append(f"{'Variable':<47} {'Source cohort':<28} Final extracted ({scenario.value})")
    lines.append("-" * 110)
    row("n (patients)", f"{source['n']:,}", f"{final['n']:,}")
    row("Records", f"{source['records']:,}", f"{final['records']:,}")
    row("% male", f"{source['male_pct']:.1f}%" if source['male_pct'] is not None else "N/A",
        f"{final['male_pct']:.1f}%" if final['male_pct'] is not None else "N/A")
    row("Mean age, years (SD)", fmt_mean_sd(source['age']), fmt_mean_sd(final['age']))
    row("Mean eGFR (SD)", fmt_mean_sd(source['egfr']) if source['egfr'] else "N/A*", fmt_mean_sd(final['egfr']))
    row("Median eGFR (IQR)", fmt_median_iqr(source['egfr']) if source['egfr'] else "N/A*", fmt_median_iqr(final['egfr']))
    if scenario != ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS:
        row("Mean uACR, mg/g (SD)", "N/A*", fmt_mean_sd(final['uacr'], decimals=1))
        row("Median uACR, mg/g (IQR)", "N/A*", fmt_median_iqr(final['uacr'], decimals=1))
    if scenario == ExperimentScenario.EIGHT_FEATURES:
        for lab in ['calcium', 'phosphate', 'bicarbonate', 'serum_albumin']:
            row(f"Mean {lab} (SD)", "N/A*", fmt_mean_sd(final['chem_panel'][lab], decimals=2))
    row("Mean follow-up, years (SD)", "N/A*", fmt_mean_sd(final['followup']))
    row("Median follow-up, years (IQR)", "N/A*", fmt_median_iqr(final['followup']))
    row("Mean time-to-ESRD, years (SD)", "N/A*", fmt_mean_sd(final['time_to_esrd']))
    row("Median time-to-ESRD, years (IQR)", "N/A*", fmt_median_iqr(final['time_to_esrd']))
    row("ESRD events (n)", f"{source['n_events']:,}", f"{final['n_events']:,}")
    row("ESRD rate, per 1,000 person-years (95% CI)", "N/A*", fmt_rate(final['incidence_rate']))
    lines.append("")
    lines.append(
        "* Not computed for the source cohort: eGFR/uACR/chem-panel/follow-up/incidence-rate for the")
    lines.append(
        "  full source population would each require a fresh labevents.csv scan (the same order of")
    lines.append(
        "  cost as the original extraction, ~2h) and were not run for reporting purposes alone.")
    lines.append(
        "  Age/gender/n/events are free (patients.csv + diagnoses_icd.csv only, no lab reads).")
    lines.append("")

    lines.append("Train/test split (final extracted cohort):")
    lines.append(
        f"  Train: {train_df['subject_id'].nunique():,} patients, {len(train_df):,} records, "
        f"{100 * train_df['has_esrd'].sum() / len(train_df):.2f}% of records positive")
    lines.append(
        f"  Test:  {test_df['subject_id'].nunique():,} patients, {len(test_df):,} records, "
        f"{100 * test_df['has_esrd'].sum() / len(test_df):.2f}% of records positive")
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
