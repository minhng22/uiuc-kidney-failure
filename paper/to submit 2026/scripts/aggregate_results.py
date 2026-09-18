#!/usr/bin/env python3
"""Build manuscript tables from saved reports, without loading model checkpoints.

Run from any directory with Python 3 (standard library only). All production
reports and expected metric triples must be complete and finite. This is an
aggregation of reported, rounded metrics, not a new evaluation or a validation
of the underlying clinical prediction protocol.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import re
import statistics


PAPER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_ROOT.parents[1]
SCENARIOS = ("four_features", "eight_features", "twenty_features_heterogeneous")
MODELS = (
    "Cox", "Dynamic DeepHit", "Hazard Transformer", "Logistic Hazard",
    "RNN-Surv", "DeepSurv", "GBSA", "Survival RF", "Survival SVM", "Weibull AFT",
)
METRICS = ("c_index", "brier", "auc")
REPS = range(1, 6)
METRIC_LINE = re.compile(r"^\s+(.+?): c_index=(\S+) brier=(\S+) auc=(\S+)\s*$")


def read_report(path, scenario, rep):
    contents = path.read_bytes()
    text = contents.decode("utf-8")
    if f"Repetition: {rep}\n" not in text or f"Scenario: {scenario}\n" not in text:
        raise ValueError(f"Report identity does not match its path: {path}")
    headings = [line for line in text.splitlines() if line.startswith("Generated on: ")]
    if len(headings) != 1:
        raise ValueError(f"Missing/ambiguous report timestamp: {path}")
    generated_on = headings[0].split(": ", 1)[1]
    sections = text.split("Discrimination metrics (", 1)
    if len(sections) != 2:
        raise ValueError(f"Missing discrimination metrics block: {path}")
    block = sections[1].split("=== Horizon:", 1)[0]
    rows = {}
    for line in block.splitlines():
        match = METRIC_LINE.match(line)
        if not match:
            continue
        model, *values = match.groups()
        if model in rows:
            raise ValueError(f"Duplicate metric row for {model}: {path}")
        numeric = [float(value) for value in values]
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError(f"Non-finite metric for {model}: {path}")
        rows[model] = dict(zip(METRICS, numeric))
    expected = set(MODELS + (("KFRE",) if scenario != SCENARIOS[-1] else ()))
    if set(rows) != expected:
        raise ValueError(f"Unexpected model coverage in {path}: {set(rows) ^ expected}")
    provenance = {
        "rep": rep,
        "scenario": scenario,
        "path": str(path.relative_to(REPO_ROOT)),
        "generated_on_as_reported": generated_on,
        "sha256": hashlib.sha256(contents).hexdigest(),
        "model_count": len(rows),
    }
    return rows, provenance


def split_fingerprint(path):
    """Audit partition identity without emitting subject identifiers."""
    subjects = {}
    rows = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            identifier = row["subject_id"]
            duration = float(row["duration_in_days"])
            event = row["has_esrd"].lower() in ("1", "1.0", "true")
            if identifier not in subjects or subjects[identifier][0] <= duration:
                subjects[identifier] = (duration, event)
    digest = hashlib.sha256("\n".join(sorted(subjects)).encode()).hexdigest()
    summary = {
        "path": str(path.relative_to(REPO_ROOT)),
        "rows": rows,
        "subjects": len(subjects),
        "last_observation_events": sum(event for _, event in subjects.values()),
        "subject_set_sha256": digest,
    }
    return set(subjects), summary


def audit_splits():
    audit = {}
    for scenario in SCENARIOS:
        partitions = {"train": [], "test": []}
        missing = []
        by_rep = {}
        for rep in REPS:
            by_rep[rep] = {}
            for split in partitions:
                path = REPO_ROOT / "generated_data" / f"rep{rep}" / f"{scenario}_{split}_data.csv"
                if not path.exists():
                    missing.append(str(path.relative_to(REPO_ROOT)))
                    continue
                identifiers, summary = split_fingerprint(path)
                by_rep[rep][split] = identifiers
                summary["rep"] = rep
                partitions[split].append(summary)
        overlap = {
            str(rep): len(parts["train"] & parts["test"])
            for rep, parts in by_rep.items() if set(parts) == {"train", "test"}
        }
        audit[scenario] = {
            "available_partitions": partitions,
            "missing_local_files": missing,
            "all_five_train_subject_sets_identical": (
                len(partitions["train"]) == 5
                and len({item["subject_set_sha256"] for item in partitions["train"]}) == 1
            ) if partitions["train"] else None,
            "all_five_test_subject_sets_identical": (
                len(partitions["test"]) == 5
                and len({item["subject_set_sha256"] for item in partitions["test"]}) == 1
            ) if partitions["test"] else None,
            "train_test_subject_overlap_by_rep": overlap,
        }
    return audit


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PAPER_ROOT / "results")
    parser.add_argument("--skip-split-audit", action="store_true",
                        help="Skip streaming locally available train/test CSVs.")
    args = parser.parse_args()
    per_run = []
    sources = []
    for scenario in SCENARIOS:
        for rep in REPS:
            path = REPO_ROOT / "generated_data" / f"rep{rep}" / f"{scenario}_clinical_validity_report.txt"
            rows, provenance = read_report(path, scenario, rep)
            sources.append(provenance)
            for model, metrics in rows.items():
                per_run.append({
                    "scenario": scenario, "model": model, "rep": rep, **metrics,
                    "source_report": provenance["path"],
                    "source_generated_on": provenance["generated_on_as_reported"],
                    "source_sha256": provenance["sha256"],
                })
    groups = defaultdict(list)
    for row in per_run:
        groups[(row["scenario"], row["model"])].append(row)
    summary = []
    for (scenario, model), rows in groups.items():
        result = {"scenario": scenario, "model": model, "n_reps": len(rows)}
        for metric in METRICS:
            values = [row[metric] for row in rows]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_sd"] = statistics.stdev(values)
            result[f"{metric}_min"] = min(values)
            result[f"{metric}_max"] = max(values)
        summary.append(result)
    assert len(per_run) == 160 and len(summary) == 32
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).relative_to(REPO_ROOT)),
        "source_reports": sources,
        "n_reports": len(sources),
        "n_per_run_rows": len(per_run),
        "n_summary_rows": len(summary),
        "repetitions": list(REPS),
        "aggregation": "Arithmetic mean and sample SD (denominator n-1) of reported rounded metrics.",
        "interpretation": "Five model repetitions on one fixed patient partition; SD is run variability, not a confidence interval or variability across resampled cohorts.",
        "metric_definitions_from_code": {
            "c_index": "Harrell concordance on per-subject labels across observed follow-up; not truncated at 730 days.",
            "brier": "Integrated Brier score on 50 time points from 1 to 730 days, using each model's scalar score with its training-fitted Breslow-style survival transform (including models with native probabilities).",
            "auc": "Mean cumulative/dynamic AUC using scalar scores at daily evaluation times 1 through 728 days, with the original training-row censoring reference.",
            "hazard_transformer_ranking_score": "Native CIF at 365 days (730-day native CIF is constrained to approximately one by the finite-support PMF).",
        },
        "local_partition_audit": None if args.skip_split_audit else audit_splits(),
        "limitations": [
            "This aggregates saved outputs without rerunning training or evaluating checkpoints.",
            "Report generation timestamps have no recorded timezone and are preserved literally.",
            "Local twenty-feature train/test CSVs may be absent; report completeness does not imply those large artifacts are locally available.",
            "Scalar-score Brier results do not directly describe calibration of native horizon probabilities.",
            "The censoring-reference training frame contains repeated observation rows whereas model predictions are per subject.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "performance_per_run.csv", per_run)
    write_csv(args.output_dir / "performance_summary.csv", summary)
    (args.output_dir / "performance_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(per_run)} per-run rows and {len(summary)} summary rows to {args.output_dir}")


if __name__ == "__main__":
    main()
