"""Patient-level terminal outcomes and patient-level comparator inputs for the
clinical-validity evaluation (PAPER_GAPS_EXPERIMENT_PLAN.md Gaps 5a/5c/5d).

Why this module exists
----------------------
Every model in pkgs/models/ produces exactly ONE prediction per test patient
(all of them either call get_last_observation_data(), which is
`groupby('subject_id').last()`, or build a Dataset whose
`list(df.groupby('subject_id'))` grouping is the same subject_id-sorted,
one-entry-per-patient order -- see pkgs/models/cox.py's module docstring for
the Stage 2.2 refactor that made this uniform). The evaluation around those
predictions, however, was still fed the RAW exported frames, which are one row
per lab EVENT:

- the IPCW censoring reference (`Surv.from_dataframe(event='has_esrd',
  time='duration_in_days', data=df_train)`) used by the integrated Brier score
  and the time-dependent AUC,
- the treat-all decision curve (durations/events straight off `df_test`),
- the eGFR-threshold referral comparator (one "decision" per eGFR row).

In those raw frames an intermediate visit of a patient still under follow-up
carries `has_esrd=0` at that visit's own `duration_in_days`, so each of the
three reads it as a TERMINAL censored observation. Measured on rep99
four_features: 5,806 training rows over 500 patients give a row-level event
rate of 5.75% against a patient-level rate of 50.0% -- an order of magnitude
of artificial censoring in the reference distribution.

Outcome semantics this module relies on (pkgs/data_analysis/time_series_store.py)
-------------------------------------------------------------------------------
- `duration_in_days` is days since that subject's FIRST lab record
  (calculate_duration_in_days), so every row of a subject shares one time
  origin and the subject's rows are ascending in it.
- ESRD-positive subjects have their records truncated at the first ESRD
  diagnosis date, and only the rows ON that date carry `has_esrd=1`
  (process_patient_esrd). So a subject's LAST row is their terminal row, and
  `max(has_esrd) == last(has_esrd)` for every subject.
- ESRD-negative subjects carry `has_esrd=0` on every row; their last row is
  administrative censoring.

Terminal outcome per patient is therefore (max duration_in_days, last
has_esrd), which is exactly the row get_last_observation_data() hands the
models. verify_patient_outcomes() asserts that correspondence instead of
assuming it.
"""
import numpy as np


def patient_level_outcomes(df):
    """One terminal (subject_id, duration_in_days, has_esrd) row per patient,
    built the same way get_last_observation_data() builds the frame the models
    are scored on: each subject's last row in ascending-duration order.

    Returns a frame sorted by subject_id (pandas' groupby default, and the
    order every model's predictions come back in)."""
    required = {'subject_id', 'duration_in_days', 'has_esrd'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"patient_level_outcomes needs columns {sorted(missing)}")
    ordered = df.sort_values(['subject_id', 'duration_in_days'], kind='mergesort')
    terminal = ordered.groupby('subject_id', as_index=False).last()
    return terminal[['subject_id', 'duration_in_days', 'has_esrd']].reset_index(drop=True)


def verify_patient_outcomes(df, terminal):
    """Checks the three properties the terminal frame is supposed to have,
    rather than trusting them (Gap 5a's "verify patient uniqueness and outcome
    consistency"). Returns a dict of findings; an empty 'problems' list means
    the terminal outcomes are safe to use as the censoring reference."""
    problems = []
    if terminal['subject_id'].duplicated().any():
        problems.append("terminal frame has duplicate subject_id rows")
    if terminal['subject_id'].nunique() != df['subject_id'].nunique():
        problems.append("terminal frame covers a different patient set than the source frame")

    grouped = df.groupby('subject_id')
    # The last row must also be the maximum-duration row, and the last row's
    # event flag must equal the subject's maximum event flag -- i.e. no subject
    # has an event recorded before their final observation (which would mean
    # the exported frame was not truncated at the event, contrary to
    # process_patient_esrd).
    max_duration = grouped['duration_in_days'].max().reset_index()
    merged = terminal.merge(max_duration, on='subject_id', suffixes=('', '_max'))
    duration_mismatch = merged.loc[
        ~np.isclose(merged['duration_in_days'], merged['duration_in_days_max']), 'subject_id']
    if len(duration_mismatch):
        problems.append(
            f"{len(duration_mismatch)} subjects whose last row is not their max-duration row")

    any_event = grouped['has_esrd'].max().reset_index().rename(columns={'has_esrd': 'has_esrd_any'})
    merged = terminal.merge(any_event, on='subject_id')
    event_mismatch = merged.loc[merged['has_esrd'] != merged['has_esrd_any'], 'subject_id']
    if len(event_mismatch):
        problems.append(
            f"{len(event_mismatch)} subjects with an event recorded before their final observation "
            "(terminal-row event flag disagrees with max event flag)")

    return {
        'n_rows': int(len(df)),
        'n_patients': int(terminal['subject_id'].nunique()),
        'row_event_rate': float(df['has_esrd'].mean()),
        'patient_event_rate': float(terminal['has_esrd'].mean()),
        'n_events': int(terminal['has_esrd'].sum()),
        'problems': problems,
    }


def align_predictions_to_patients(terminal, durations, events, model_name, tolerance_days=1.0):
    """Confirms a model's returned (durations, events) really are this
    scenario's patient-level terminal outcomes, in the canonical subject_id
    order, before its predictions are paired with anything patient-level.

    Every model's predictions() derives its own durations/events from either
    get_last_observation_data() or a `groupby('subject_id')` Dataset, so they
    are the same patients in the same order. They are not always the same
    NUMBERS, though, and the differences are real:

    - Dynamic-DeepHit and Hazard Transformer floor each duration to a whole day
      (both discretize time into day-indexed bins), so their durations sit up to
      one day below the exact fractional duration — 274 of rep99
      four_features' 345 test patients, max difference 0.999 d.
    - Weibull AFT replaces a duration of exactly 0 with 1e-5 (lifelines errors
      on a zero duration).

    Neither changes which patient is which, but it does mean those models were
    being scored against slightly different outcomes than the other nine. The
    caller therefore scores EVERY model against `terminal` and uses each model
    only for its risk scores and its native survival curve; this function
    decides whether that substitution is safe (same patients, same events,
    sub-`tolerance_days` timing differences) and reports what the differences
    were.

    Returns (ok, message)."""
    durations = np.asarray(durations, dtype=np.float64)
    events = np.asarray(np.asarray(events)).astype(float)
    if len(durations) != len(terminal):
        return False, (f"{model_name}: {len(durations)} predictions vs "
                       f"{len(terminal)} test patients — not patient-aligned")

    reference_durations = terminal['duration_in_days'].values.astype(float)
    reference_events = terminal['has_esrd'].values.astype(int)
    max_diff = float(np.max(np.abs(durations - reference_durations)))
    n_diff = int((np.abs(durations - reference_durations) > 1e-6).sum())
    events_equal = np.array_equal(events.astype(int), reference_events)

    if not events_equal:
        n_event_diff = int((events.astype(int) != reference_events).sum())
        return False, (f"{model_name}: {n_event_diff} event flags differ from the patient-level "
                       "terminal outcomes — not the same outcome set")
    if max_diff > tolerance_days:
        return False, (f"{model_name}: durations differ from the patient-level terminal outcomes "
                       f"by up to {max_diff:.4f} d (> {tolerance_days} d tolerance)")
    if n_diff:
        return True, (f"{model_name}: aligned to {len(terminal)} patient-level terminal outcomes "
                      f"({n_diff} durations differ by up to {max_diff:.6g} d — this model's own "
                      "time handling (day-bin flooring, or a zero-duration floor); scored against "
                      "the canonical outcomes)")
    return True, f"{model_name}: aligned to {len(terminal)} patient-level terminal outcomes"


def patient_level_egfr(df, terminal):
    """One eGFR-based referral decision per patient, at that patient's own
    prediction landmark (their last observation -- the same row every model is
    scored on), for the eGFR-threshold comparator (Gap 5d).

    Selection rule: the patient's most recent GENUINELY MEASURED eGFR at or
    before the landmark. In twenty_features_heterogeneous each exported row
    carries only the one lab that was actually drawn, with every other lab set
    to a placeholder 0 plus a `<lab>_missing=1` flag, so rows with
    `egfr_missing == 1` carry no eGFR at all and are skipped; the landmark row
    itself is frequently one of them. four_features/eight_features have no
    `egfr_missing` column (every row is anchored on a real creatinine draw), so
    the rule reduces to "the landmark row's own eGFR" there.

    Returns (frame, info). `frame` has subject_id/egfr/duration_in_days/
    has_esrd for the patients that HAVE a usable eGFR, carrying the terminal
    outcome (not the eGFR row's own duration). `info` records how many patients
    were dropped for having no measured eGFR, so the report can state the
    comparator's denominator explicitly instead of silently comparing rules on
    different patient sets."""
    if 'egfr' not in df.columns:
        return None, {'reason': 'scenario has no egfr column'}

    candidates = df[['subject_id', 'duration_in_days', 'egfr']].copy()
    n_rows_total = len(candidates)
    if 'egfr_missing' in df.columns:
        candidates = candidates[df['egfr_missing'] == 0]
    landmark = terminal.set_index('subject_id')['duration_in_days']
    candidates = candidates[
        candidates['duration_in_days'] <= candidates['subject_id'].map(landmark) + 1e-9]

    candidates = candidates.sort_values(['subject_id', 'duration_in_days'], kind='mergesort')
    selected = candidates.groupby('subject_id', as_index=False).last()[['subject_id', 'egfr']]

    frame = terminal.merge(selected, on='subject_id', how='inner').reset_index(drop=True)
    info = {
        'rows_considered': int(n_rows_total),
        'rows_with_measured_egfr': int(len(candidates)),
        'patients_total': int(len(terminal)),
        'patients_with_egfr': int(len(frame)),
        'patients_without_egfr': int(len(terminal) - len(frame)),
    }
    return frame, info
