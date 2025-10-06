#!/usr/bin/env python3
"""Check per-subject eGFR measurement span in generated CSVs.

Scans generated_data/rep*/egfr_tv_{train,test}_data.csv, computes span = max(stop)-min(start)
and reports patients with span < threshold (default 90 days). Writes per-file CSVs listing failing subjects.
"""
import glob
import os
import sys
from pathlib import Path
import argparse

THRESHOLD_DAYS = 90.0


def analyze_file(path, threshold=THRESHOLD_DAYS):
    try:
        import pandas as pd
    except Exception:
        print("pandas is required to run this script. Please install it (pip install pandas)")
        sys.exit(2)

    df = pd.read_csv(path)
    cols = set(df.columns)
    # Prefer duration_in_days if present (it's an elapsed time since baseline), otherwise fall back to start/stop
    if "duration_in_days" in cols:
        # aggregate per-subject; include an ESRD flag if present
        if "has_esrd" in cols:
            grp = df.groupby("subject_id").agg(
                min_d=("duration_in_days", "min"),
                max_d=("duration_in_days", "max"),
                n_obs=("egfr", "count"),
                has_esrd_any=("has_esrd", "max"),
            )
        else:
            grp = df.groupby("subject_id").agg(min_d=("duration_in_days", "min"), max_d=("duration_in_days", "max"), n_obs=("egfr", "count"))
        grp = grp.reset_index()
        grp["span_days"] = grp["max_d"] - grp["min_d"]
    else:
        raise ValueError("File does not contain 'duration_in_days' column.")

    total = len(grp)
    ok_mask = grp["span_days"] >= threshold
    ok = int(ok_mask.sum())
    fail_mask = grp["span_days"] < threshold
    fail = int(fail_mask.sum())

    # count how many failing subjects have ESRD (subject-level any)
    fail_with_esrd = 0
    fail_with_esrd_pct = 0.0
    ok_with_esrd = 0
    ok_with_esrd_pct = 0.0
    if "has_esrd_any" in grp.columns:
        # ensure numeric
        fail_with_esrd = int(grp.loc[fail_mask, "has_esrd_any"].fillna(0).astype(int).sum())
        ok_with_esrd = int(grp.loc[ok_mask, "has_esrd_any"].fillna(0).astype(int).sum())
        if fail > 0:
            fail_with_esrd_pct = 100.0 * fail_with_esrd / int(fail)
        if ok > 0:
            ok_with_esrd_pct = 100.0 * ok_with_esrd / int(ok)

    print(f"File: {path}")
    print(f"  total subjects: {total}")
    # print >= threshold line with ESRD info if available
    if ok > 0 and ok_with_esrd > 0:
        print(f"  subjects with span >= {threshold} days: {ok} (of which with ESRD: {ok_with_esrd}, {ok_with_esrd_pct:.1f}% )")
    else:
        print(f"  subjects with span >= {threshold} days: {ok} (of which with ESRD: {ok_with_esrd})")

    if fail > 0 and fail_with_esrd > 0:
        print(f"  subjects with span <  {threshold} days: {fail} (of which with ESRD: {fail_with_esrd}, {fail_with_esrd_pct:.1f}% )")
    else:
        print(f"  subjects with span <  {threshold} days: {fail} (of which with ESRD: {fail_with_esrd})")

    # write failing subjects to CSV next to the file
    outpath = Path(path).with_name(Path(path).stem + f"_span_lt_{int(threshold)}.csv")
    # choose which min/max column names to output depending on which method was used
    if "duration_in_days" in cols:
        out_cols = ["subject_id", "n_obs", "min_d", "max_d", "span_days"]
        if "has_esrd_any" in grp.columns:
            out_cols.append("has_esrd_any")
    else:
        out_cols = ["subject_id", "n_obs", "min_start", "max_stop", "span_days"]

    # write subjects with span < threshold
    outpath_lt = Path(path).with_name(Path(path).stem + f"_span_lt_{int(threshold)}.csv")
    grp.loc[grp["span_days"] < threshold, out_cols].to_csv(outpath_lt, index=False)
    print(f"  wrote subjects with span < {threshold} to: {outpath_lt}")

    # also write subjects with span >= threshold
    outpath_ge = Path(path).with_name(Path(path).stem + f"_span_ge_{int(threshold)}.csv")
    grp.loc[grp["span_days"] >= threshold, out_cols].to_csv(outpath_ge, index=False)
    print(f"  wrote subjects with span >= {threshold} to: {outpath_ge}\n")
    return {
        "file": path,
        "total": total,
        "ok": int(ok),
        "fail": int(fail),
        "fail_with_esrd": int(fail_with_esrd),
        "ok_with_esrd": int(ok_with_esrd),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check per-subject eGFR span in generated CSVs.")
    parser.add_argument("--reps", nargs="*", default=["rep5"],
                        help="Which rep directories to check (default: rep1). Example: --reps rep1 rep2")
    parser.add_argument("--all", action="store_true", help="Check all reps (overrides --reps)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_DAYS, help="Threshold in days (default 90)")
    args = parser.parse_args(argv)

    # We will combine per-rep train + test files first (if present), write a combined CSV
    # and then analyze the combined file. If only one of train/test exists, analyze that file.
    reps_to_check = []
    if args.all:
        # find rep directories under generated_data
        rep_dirs = sorted(glob.glob(os.path.join("generated_data", "rep*")))
        reps_to_check = [os.path.basename(d) for d in rep_dirs]
    else:
        reps_to_check = args.reps

    results = []
    any_found = False
    for rep in reps_to_check:
        rep_dir = os.path.join("generated_data", rep)
        if not os.path.isdir(rep_dir):
            print(f"Skipping missing rep directory: {rep_dir}")
            continue
        # strictly read the canonical train/test files, combine them, and analyze the combined file
        train_file = os.path.join(rep_dir, "egfr_tv_train_data.csv")
        test_file = os.path.join(rep_dir, "egfr_tv_test_data.csv")
        if not os.path.isfile(train_file) or not os.path.isfile(test_file):
            print(f"Skipping {rep}: both train and test files required.\n  train present: {os.path.isfile(train_file)}\n  test present:  {os.path.isfile(test_file)}")
            continue
        any_found = True

        try:
            import pandas as pd
        except Exception:
            print("pandas is required to run this script. Please install it (pip install pandas)")
            sys.exit(2)

        print(f"Combining train+test for rep {rep}:\n  train: {train_file}\n  test:  {test_file}")
        try:
            df_train = pd.read_csv(train_file)
            df_test = pd.read_csv(test_file)
        except Exception as e:
            print(f"Failed to read train/test for {rep}: {e}")
            continue

        combined_df = pd.concat([df_train, df_test], ignore_index=True)
        combined_path = os.path.join(rep_dir, "egfr_tv_combined_data.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"  wrote combined file: {combined_path}")

        res = analyze_file(combined_path, threshold=args.threshold)
        if res:
            results.append(res)

    # summary across selected files
    if results:
        tot_subjects = sum(r["total"] for r in results)
        tot_ok = sum(r["ok"] for r in results)
        tot_fail = sum(r["fail"] for r in results)
        tot_fail_esrd = sum(r.get("fail_with_esrd", 0) for r in results)
        tot_ok_esrd = sum(r.get("ok_with_esrd", 0) for r in results)

        # percentages (guard against division by zero)
        tot_ok_pct = (100.0 * tot_ok / tot_subjects) if tot_subjects > 0 else 0.0
        tot_fail_pct = (100.0 * tot_fail / tot_subjects) if tot_subjects > 0 else 0.0
        tot_fail_esrd_pct = (100.0 * tot_fail_esrd / tot_fail) if tot_fail > 0 else 0.0
        tot_ok_esrd_pct = (100.0 * tot_ok_esrd / tot_ok) if tot_ok > 0 else 0.0

        print("Overall summary:")
        print(f"  total subjects: {tot_subjects}")
        print(f"  subjects with span >= {args.threshold} days: {tot_ok} ({tot_ok_pct:.1f}%) , of which with ESRD: {tot_ok_esrd} ({tot_ok_esrd_pct:.1f}%)")
        print(f"  subjects with span <  {args.threshold} days: {tot_fail} ({tot_fail_pct:.1f}%) , of which with ESRD: {tot_fail_esrd} ({tot_fail_esrd_pct:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
