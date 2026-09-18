"""One demographic record per patient, for subgroup performance analysis
(PAPER_GAPS_EXPERIMENT_PLAN.md Gap 12).

Gap 12 asks for this to "reuse the existing demographic utilities", but also to
"audit race handling before reuse" and to "define age at the prediction landmark
explicitly rather than treating `anchor_age` as landmark age". Both audits
changed what could be reused, so the selection rules live here rather than in
pkgs/data_analysis/demographics.py, whose functions exist to PRINT cohort
distributions and whose filtering is appropriate for that purpose but not for
this one:

Race. `store.get_admission_df()` drops every admission whose race is
"PATIENT DECLINED TO ANSWER" / "UNABLE TO OBTAIN" / "UNKNOWN" before doing
anything else, and `demographics.ethnicity_and_race_statistics()` then keeps the
FIRST remaining admission row per patient. For a distribution table that is a
defensible choice. For subgroup PERFORMANCE it is not: patients whose race was
never recorded would silently vanish from the denominator, so the subgroup rows
would no longer sum to the cohort and "missing" would be invisible rather than
reported. This module keeps those patients under an explicit
`UNKNOWN/NOT RECORDED` group. It also replaces "first admission row" with "most
frequently recorded value across the patient's admissions, ties broken by the
earliest admission" — a patient with several admissions can have several
recorded values, and taking whichever happened to sort first is arbitrary in a
way that a modal rule is not. Both rules are reported alongside the numbers.

Age. `patients.csv` gives `anchor_age`, the patient's age in `anchor_year` —
NOT their age at the prediction landmark. The exported scenario CSVs carry
`duration_in_days` (days since that patient's FIRST lab record) but no absolute
timestamp, so the calendar distance between `anchor_year` and the first lab
record cannot be recovered from them, and landmark age is therefore not
derivable from the data this analysis has. Rather than quietly relabel
`anchor_age` as landmark age, this module returns `anchor_age` under its own
name, plus `years_from_first_record_to_landmark`, so the report can state the
grouping variable exactly and quantify how far the landmark sits from the
anchor. The four-/eight-feature scenario CSVs' own `age` column is that same
`anchor_age` (time_series_store.py renames it), constant across a patient's
rows, so the two agree by construction.
"""
import pandas as pd

from pkgs.commons import patients_file_path, admissions_file_path, age_bins
from pkgs.data_analysis.store import get_admission_df

AGE_GROUP_LABELS = ['<27', '27-54', '54-82', '82+']
UNKNOWN_RACE = 'UNKNOWN/NOT RECORDED'
_UNRECORDED_RACE_VALUES = ("PATIENT DECLINED TO ANSWER", "UNABLE TO OBTAIN", "UNKNOWN")


def _patient_level_race(subject_ids):
    """Modal recorded race per patient, ties broken by earliest admission, with
    an explicit UNKNOWN/NOT RECORDED group for patients who have no recorded
    value (either no admission row at all, or only unrecorded ones).

    get_admission_df(True) is reused for the ethnicity->race collapsing so the
    category names match the rest of the repo's demographic reporting, but the
    patients it drops are recovered here rather than lost."""
    mapped = get_admission_df(ethnicity_to_race=True)
    mapped = mapped[mapped['subject_id'].isin(subject_ids)][['subject_id', 'hadm_id', 'admittime', 'race']]

    all_admissions = pd.read_csv(admissions_file_path, usecols=['subject_id', 'hadm_id', 'race'])
    all_admissions = all_admissions[all_admissions['subject_id'].isin(subject_ids)]
    n_unrecorded_rows = int(all_admissions['race'].isin(_UNRECORDED_RACE_VALUES).sum())

    mapped = mapped.sort_values(['subject_id', 'admittime'], kind='mergesort')
    mapped['order'] = mapped.groupby('subject_id').cumcount()
    counts = (mapped.groupby(['subject_id', 'race'])
              .agg(n=('race', 'size'), first_order=('order', 'min'))
              .reset_index()
              .sort_values(['subject_id', 'n', 'first_order'], ascending=[True, False, True],
                           kind='mergesort'))
    selected = counts.groupby('subject_id', as_index=False).first()[['subject_id', 'race']]

    race = pd.DataFrame({'subject_id': pd.Series(sorted(set(subject_ids)))})
    race = race.merge(selected, on='subject_id', how='left')
    n_multi = int((counts.groupby('subject_id').size() > 1).sum())
    n_missing = int(race['race'].isna().sum())
    race['race'] = race['race'].fillna(UNKNOWN_RACE)
    info = {
        'patients': int(len(race)),
        'patients_with_recorded_race': int(len(selected)),
        'patients_without_recorded_race': n_missing,
        'patients_with_more_than_one_recorded_race': n_multi,
        'admission_rows_with_unrecorded_race': n_unrecorded_rows,
        'selection_rule': ('most frequently recorded race across the patient\'s admissions, '
                           'ties broken by earliest admission; patients with no recorded value '
                           f'are kept as {UNKNOWN_RACE}'),
    }
    return race, info


def patient_metadata(terminal):
    """Demographic record per held-out patient, joined onto that patient's
    terminal outcome (patient_outcomes.patient_level_outcomes()).

    Returns (frame, info). The frame carries subject_id, gender, anchor_age,
    age_group, race, plus duration_in_days/has_esrd from `terminal` and
    `years_from_first_record_to_landmark`. `info` records the join coverage and
    the race-selection audit, both of which the report prints verbatim so the
    grouping rules are visible next to the numbers they produced."""
    subject_ids = terminal['subject_id'].unique()
    patients = pd.read_csv(patients_file_path,
                           usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year'])
    patients = patients[patients['subject_id'].isin(subject_ids)]

    frame = terminal.merge(patients, on='subject_id', how='left')
    n_missing_patient_row = int(frame['anchor_age'].isna().sum())

    race, race_info = _patient_level_race(subject_ids)
    frame = frame.merge(race, on='subject_id', how='left')
    frame['race'] = frame['race'].fillna(UNKNOWN_RACE)
    frame['gender'] = frame['gender'].fillna('UNKNOWN')
    frame['age_group'] = pd.cut(frame['anchor_age'], bins=age_bins,
                                labels=AGE_GROUP_LABELS, right=False)
    frame['age_group'] = frame['age_group'].cat.add_categories(['UNKNOWN']).fillna('UNKNOWN')
    frame['years_from_first_record_to_landmark'] = frame['duration_in_days'] / 365.25
    # The analyzer indexes model predictions by this frame's positional index, so
    # it must stay 0..n-1 in `terminal`'s order. Both merges above are 1:1 left
    # joins (patients.csv and the race frame each carry one row per subject_id)
    # and pandas returns a fresh RangeIndex, so this is already true — asserted
    # rather than assumed, because a silent violation would pair one patient's
    # risk score with another patient's group.
    frame = frame.reset_index(drop=True)
    assert frame['subject_id'].tolist() == terminal['subject_id'].tolist(), \
        "patient_metadata reordered or duplicated patients relative to the terminal outcomes"

    info = {
        'patients': int(len(frame)),
        'patients_missing_from_patients_csv': n_missing_patient_row,
        'race': race_info,
        'age_variable': ('anchor_age from patients.csv — the patient\'s age in their MIMIC anchor '
                         'year, NOT their age at the prediction landmark; the exported scenario '
                         'CSVs carry no absolute timestamp, so landmark age is not derivable here'),
        'landmark_offset_years': {
            'median': round(float(frame['years_from_first_record_to_landmark'].median()), 3),
            'p90': round(float(frame['years_from_first_record_to_landmark'].quantile(0.9)), 3),
            'max': round(float(frame['years_from_first_record_to_landmark'].max()), 3),
        },
    }
    return frame, info


def subgroup_definitions():
    """The (column, human label) pairs subgroup performance is reported over.
    Kept in one place so the analyzer and the report agree on both the set and
    the wording."""
    return [
        ('age_group', 'Age group (anchor_age bins — see patient_metadata: NOT landmark age)'),
        ('gender', 'Sex as recorded in patients.csv'),
        ('race', 'Race (collapsed from admissions.csv ethnicity; unknown retained)'),
    ]
