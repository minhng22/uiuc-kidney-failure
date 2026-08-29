"""
One-off trial driver for finding the largest patient-subsample of
TWENTY_FEATURES_HETEROGENEOUS's training data that CoxTimeVaryingFitter.fit()
can actually complete on, per user request (2026-08-29): "Subsample the
training data. Descend the sample size slowly. See the biggest sample size
you can fit in" -- Cox's full-data fit (26,080 train patients, 6.51M rows,
~250 rows/patient) reliably OOMs (confirmed twice: a real numpy
ArrayMemoryError after ~2h on the original run, a hard SIGKILL after 1h43m
on an isolated batched retry with ~170GB free) -- see
generated_data/rep1/stage3_0_background_process_log.txt for the full
diagnosis. Not wired into the normal training path -- standalone, for this
search only. Usage:
    python -m pkgs.scripts.cox_twenty_subsample_trial <n_patients> [seed]
Exit 0 on a completed fit, exit 1 on any exception (including a caught
MemoryError). Prints timing on success. A separate bash wrapper
(cox_twenty_subsample_search.sh) polls this process's RSS and kills it
early (recorded as a "too-big" failure, not making the OS actually OOM-kill
the whole box) if it crosses a safety cap, to keep each trial's wall time
bounded during the search.
"""
import sys
import time

from lifelines import CoxTimeVaryingFitter

from pkgs.data_analysis.types import ExperimentScenario
from pkgs.data_analysis.model_data_store import get_train_test_data


def main():
    if len(sys.argv) not in (2, 3):
        print("Usage: python -m pkgs.scripts.cox_twenty_subsample_trial <n_patients> [seed]")
        sys.exit(2)
    n_patients = int(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv) == 3 else 42

    print(f"Loading TWENTY_FEATURES_HETEROGENEOUS train/test data...", flush=True)
    t0 = time.time()
    df_train, _df_test = get_train_test_data(ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS)
    print(f"Loaded full train: {len(df_train)} rows, "
          f"{df_train['subject_id'].nunique()} unique patients "
          f"({time.time() - t0:.1f}s)", flush=True)

    # Per-patient event label = that patient's LAST row's has_esrd (matches
    # how the rest of this codebase treats has_esrd for this counting-process
    # format -- see stage2_2_debug_report.txt Finding #3).
    last_rows = df_train.sort_values('duration_in_days').groupby('subject_id').tail(1)
    all_ids = last_rows['subject_id'].values
    labels = last_rows.set_index('subject_id')['has_esrd']

    if n_patients >= len(all_ids):
        sampled_ids = all_ids
    else:
        pos_ids = last_rows.loc[last_rows['has_esrd'] == 1, 'subject_id'].values
        neg_ids = last_rows.loc[last_rows['has_esrd'] == 0, 'subject_id'].values
        pos_frac = len(pos_ids) / len(all_ids)
        n_pos = max(1, round(n_patients * pos_frac))
        n_neg = max(1, n_patients - n_pos)
        import numpy as np
        rng = np.random.default_rng(seed)
        sampled_pos = rng.choice(pos_ids, size=min(n_pos, len(pos_ids)), replace=False)
        sampled_neg = rng.choice(neg_ids, size=min(n_neg, len(neg_ids)), replace=False)
        sampled_ids = np.concatenate([sampled_pos, sampled_neg])

    df_sub = df_train[df_train['subject_id'].isin(sampled_ids)].copy()
    n_events = labels.loc[sampled_ids].sum()
    print(f"Subsampled train: {len(df_sub)} rows, {len(sampled_ids)} patients "
          f"({n_events} events, {n_events/len(sampled_ids)*100:.1f}% event rate), "
          f"seed={seed}", flush=True)

    print("Fitting CoxTimeVaryingFitter...", flush=True)
    t_fit0 = time.time()
    model = CoxTimeVaryingFitter(penalizer=1.0)
    model.fit(df_sub, event_col='has_esrd', id_col='subject_id')
    fit_time = time.time() - t_fit0
    print(f"SUCCESS: fit completed in {fit_time:.1f}s "
          f"({len(sampled_ids)} patients, {len(df_sub)} rows)", flush=True)


if __name__ == '__main__':
    main()
