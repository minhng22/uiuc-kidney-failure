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
    parser.add_argument("--reps", nargs="*", default=["rep1"],
                        help="Which rep directories to check (default: rep1). Example: --reps rep1 rep2")
    parser.add_argument("--all", action="store_true", help="Check all reps (overrides --reps)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_DAYS, help="Threshold in days (default 90)")
    args = parser.parse_args(argv)

    if args.all:
        base_glob = os.path.join("generated_data", "rep*", "egfr_tv_*_data.csv")
        files = sorted(glob.glob(base_glob))
    else:
        files = []
        for rep in args.reps:
            files.extend(sorted(glob.glob(os.path.join("generated_data", rep, "egfr_tv_*_data.csv"))))

    if not files:
        print("No files found to check. Use --all or specify --reps. Checked reps:", args.reps)
        return 1

    results = []
    for f in files:
        res = analyze_file(f, threshold=args.threshold)
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
