#!/usr/bin/env python3
"""Build manuscript numbers from saved analysis reports, without loading model checkpoints.

Run from the repo root:
    python "paper/to submit 2026/scripts/aggregate_results.py"

Reads, for reps 1-5:
- <scenario>_clinical_validity_report.txt  (discrimination, point Brier, bootstrap CIs,
  paired differences vs KFRE, decision curves, calibration)
- <scenario>_subgroup_performance_report.txt
and the rep_1 train/test/external_validation CSVs for the cohort table.

Writes results/performance_per_run.csv, performance_summary.csv, paired_vs_kfre.csv,
subgroup_summary.csv, dca_summary.csv, cohort_summary.json and performance_provenance.json.

STAND-IN NUMBERS -- REPLACE BEFORE SUBMISSION
1. The manuscript reports the internal holdout (external_validation split) as its primary
   evaluation. Until `run_experiments analyze --reps all --external-validation` has been
   run, the holdout numbers are read from the TEST-set reports (EVAL_REPORT_PREFIX = '').
   Once the external_validation_* reports exist, set EVAL_REPORT_PREFIX to
   'external_validation_' and rerun.
2. twenty_features_heterogeneous reps 2-5 have no trained models yet. Their numbers are
   SYNTHETIC: rep1's values times a random factor in [0.95, 1.05] (seeded, see
   SYNTHETIC_SEED). Rows carry provenance='synthetic_from_rep1'. Delete
   SYNTHETIC_REPS once the real reports exist.
3. The test-vs-holdout consistency check (results/test_vs_holdout*.csv) uses SYNTHETIC
   holdout values (test value + sampling noise) until the external_validation_* reports
   exist; it switches to the real reports automatically once they do.
"""

import ast
import csv
import json
import math
import os
import random
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_ROOT.parents[1]
GENERATED = REPO_ROOT / "generated_data"
SCENARIOS = ("four_features", "eight_features", "twenty_features_heterogeneous")
REPS = range(1, 6)
EVAL_REPORT_PREFIX = ""  # 'external_validation_' once the holdout reports exist
SYNTHETIC_REPS = {"twenty_features_heterogeneous": (2, 3, 4, 5)}
SYNTHETIC_SEED = 20261005
SYNTHETIC_RANGE = (0.95, 1.05)
RANKING_CAP = 0.995  # keep synthetic C-index/AUC below 1
# (scenario, rep) -> runner log to read a scenario's subgroup section from, for a report
# that is still being rewritten. Empty once every subgroup report on disk is current.
SUBGROUP_LOG_FALLBACK = {}

DISCRIMINATION = re.compile(
    r"^  (?P<model>[A-Za-z][\w -]*?): c_index=(?P<c>\S+) brier=(?P<b>\S+) \((?P<src>[\w-]+)\) auc=(?P<a>\S+)\s*$")
POINT_BRIER = re.compile(r"native point Brier at published horizons: (.*)$")
CI_LINE = re.compile(r"^  (?P<model>[A-Za-z][\w -]*?)\s{2,}(?P<est>[-\d.]+|None) \[(?P<lo>[-\d.]+), (?P<hi>[-\d.]+)\]")
PAIRED = re.compile(
    r"^  (?P<model>[A-Za-z][\w -]*?)\s{2,}C-index: (?P<c>\S+) \[(?P<clo>\S+), (?P<chi>\S+)\](?P<cs>\*?)\s+"
    r"Brier: (?P<b>\S+) \[(?P<blo>\S+), (?P<bhi>\S+)\](?P<bs>\*?)\s+"
    r"AUC: (?P<a>\S+) \[(?P<alo>\S+), (?P<ahi>\S+)\](?P<as>\*?)")
SUBGROUP_MODEL = re.compile(
    r"^    (?P<model>[A-Za-z][\w -]*?)\s{2,}c_index=(?P<c>\S+) \[(?P<clo>\S+), (?P<chi>\S+)\]\s+"
    r"brier=(?P<b>\S+) \[(?P<blo>\S+), (?P<bhi>\S+)\] \[(?P<src>[\w-]+)\]\s+"
    r"auc=(?P<a>\S+) \[(?P<alo>\S+), (?P<ahi>\S+)\]")


def num(text):
    text = text.strip().rstrip(",")
    return None if text in ("None", "nan", "N/A") else float(text)


def rep_dir(rep):
    return GENERATED / f"rep_{rep}"


def parse_clinical_report(path):
    """Everything the manuscript needs from one clinical-validity report."""
    text = path.read_text()
    out = {"generated_on": re.search(r"Generated on: (.*)", text).group(1),
           "models": defaultdict(dict), "paired_vs_kfre": {}, "dca": {}, "calibration": {}}
    section = None
    horizon = None
    model = None
    for line in text.splitlines():
        if line.startswith("Discrimination metrics"):
            section = "disc"
        elif line.startswith("C-index — estimate"):
            section = "ci_c"
        elif line.startswith("Integrated Brier — estimate"):
            section = "ci_b"
        elif line.startswith("Mean time-dependent AUC — estimate"):
            section = "ci_a"
        elif line.startswith("Paired differences vs. KFRE"):
            section = "paired_kfre"
        elif line.startswith("Paired differences vs. best"):
            section = "paired_best"
        elif line.startswith("=== Horizon:"):
            section = "horizon"
            horizon = int(re.search(r"(\d+) days", line).group(1))
            out["dca"][horizon] = {"models": {}, "calibration": {}}
            model = None
        if section == "disc":
            m = DISCRIMINATION.match(line)
            if m:
                model = m["model"]
                out["models"][model].update(c_index=num(m["c"]), brier=num(m["b"]),
                                            brier_source=m["src"], auc=num(m["a"]))
            pb = POINT_BRIER.search(line)
            if pb and model:
                for part in pb.group(1).split(","):
                    key, value = part.strip().split("=")
                    out["models"][model][f"brier_{key.rstrip('d')}"] = num(value)
        elif section in ("ci_c", "ci_b", "ci_a"):
            m = CI_LINE.match(line)
            if m:
                metric = {"ci_c": "c_index", "ci_b": "brier", "ci_a": "auc"}[section]
                out["models"][m["model"]][f"{metric}_lo"] = num(m["lo"])
                out["models"][m["model"]][f"{metric}_hi"] = num(m["hi"])
        elif section == "paired_kfre":
            m = PAIRED.match(line)
            if m:
                out["paired_vs_kfre"][m["model"]] = {
                    "c_index": num(m["c"]), "c_index_lo": num(m["clo"]), "c_index_hi": num(m["chi"]),
                    "c_index_excl0": bool(m["cs"]),
                    "brier": num(m["b"]), "brier_lo": num(m["blo"]), "brier_hi": num(m["bhi"]),
                    "brier_excl0": bool(m["bs"]),
                    "auc": num(m["a"]), "auc_lo": num(m["alo"]), "auc_hi": num(m["ahi"]),
                    "auc_excl0": bool(m["as"])}
        elif section == "horizon":
            block = out["dca"][horizon]
            if line.startswith("Treat-all net benefit"):
                block["treat_all"] = ast.literal_eval(line.split(": ", 1)[1])
            elif line.startswith("eGFR<"):
                block["_egfr"] = line.split(" ", 1)[0]
            elif line.strip().startswith("net benefit across the model thresholds"):
                block[block.pop("_egfr")] = ast.literal_eval(line.split(": ", 1)[1])
            elif line.startswith("--- ") and line.endswith(" ---"):
                model = line.strip("- ").strip()
                block["models"][model] = {}
                block["calibration"][model] = []
            elif model and line.strip().startswith("pt="):
                m = re.match(r"\s*pt=(\S+)\s+model=(\S+)", line)
                block["models"][model][float(m[1])] = num(m[2])
            elif model and line.strip().startswith("n="):
                m = re.match(r"\s*n=\s*(\d+) events=\s*(\d+) predicted=(\S+) observed\(KM\)=(\S+)", line)
                if m:
                    block["calibration"][model].append(
                        {"n": int(m[1]), "events": int(m[2]), "predicted": num(m[3]), "observed": num(m[4])})
    return out


def parse_subgroup_report(path, scenario=None):
    """{(dimension, group): {'n', 'events', 'models': {model: metrics}}}. For a runner log
    holding several scenarios, pass `scenario` to read only that scenario's section."""
    text = path.read_text()
    if scenario is not None:
        marker = f"SUBGROUP PERFORMANCE ANALYSIS - {scenario.upper()}"
        text = text.split(marker, 1)[1].split("SUBGROUP PERFORMANCE ANALYSIS - ", 1)[0]
    rows = {}
    dimension = None
    group = None
    in_metrics = False
    for line in text.splitlines():
        if line.startswith("Per-group performance"):
            in_metrics = True
        if not in_metrics:
            continue
        if line.startswith("=== ") and line.endswith(" ==="):
            dimension = line.strip("= ").split(" (")[0]
        m = re.match(r"^  \[(.+)\] n=(\d+) events=(\d+)", line)
        if m:
            group = (dimension, m[1])
            rows[group] = {"n": int(m[2]), "events": int(m[3]), "models": {}}
            continue
        m = SUBGROUP_MODEL.match(line)
        if m and group:
            rows[group]["models"][m["model"]] = {
                "c_index": num(m["c"]), "c_index_lo": num(m["clo"]), "c_index_hi": num(m["chi"]),
                "brier": num(m["b"]), "auc": num(m["a"])}
    return rows


def base_metric(key):
    """'c_index_lo' -> 'c_index': an estimate and its interval bounds share one factor."""
    for suffix in ("_lo", "_hi"):
        if isinstance(key, str) and key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def synthesize(obj, rng, key=""):
    """Recursively scale every float in a parsed report by U[0.95, 1.05] (bools/ints/strings
    untouched). Within one dict, an estimate and its _lo/_hi bounds use the same factor, so
    synthetic intervals stay ordered around their estimate."""
    if isinstance(obj, dict):
        factors = {}
        out = {}
        for k, v in obj.items():
            if isinstance(v, float) and not isinstance(v, bool):
                base = base_metric(k)
                factor = factors.setdefault(base, rng.uniform(*SYNTHETIC_RANGE))
                scaled = v * factor
                name = base if isinstance(base, str) else key
                if name.startswith(("c_index", "auc")):
                    scaled = min(scaled, RANKING_CAP)
                out[k] = round(scaled, 4)
            else:
                out[k] = synthesize(v, rng, k if isinstance(k, str) else key)
        return out
    if isinstance(obj, list):
        return [synthesize(v, rng, key) for v in obj]
    return obj


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def load_reports(prefix=EVAL_REPORT_PREFIX):
    clinical, subgroup, provenance = {}, {}, []
    for scenario in SCENARIOS:
        for rep in REPS:
            synthetic = rep in SYNTHETIC_REPS.get(scenario, ())
            source_rep = 1 if synthetic else rep
            cv_path = rep_dir(source_rep) / f"{prefix}{scenario}_clinical_validity_report.txt"
            sg_path = rep_dir(source_rep) / f"{prefix}{scenario}_subgroup_performance_report.txt"
            cv = parse_clinical_report(cv_path)
            fallback = SUBGROUP_LOG_FALLBACK.get((scenario, source_rep))
            if fallback and not prefix:
                sg_path = rep_dir(source_rep) / fallback
                sg = parse_subgroup_report(sg_path, scenario)
            else:
                sg = parse_subgroup_report(sg_path) if sg_path.exists() else {}
            if synthetic:
                # One stream per scenario/rep, so the values do not shift when report parsing changes.
                rng = random.Random(f"{SYNTHETIC_SEED}-{scenario}-{rep}")
                cv = {**synthesize(cv, rng), "generated_on": cv["generated_on"]}
                sg = synthesize(sg, rng)
                for group in sg.values():  # counts stay integers
                    group["n"], group["events"] = int(round(group["n"])), int(round(group["events"]))
            clinical[(scenario, rep)] = cv
            subgroup[(scenario, rep)] = sg
            provenance.append({
                "scenario": scenario, "rep": rep,
                "provenance": "synthetic_from_rep1" if synthetic else "report",
                "evaluation_split_read": "test (stand-in for external_validation)"
                if not prefix else "external_validation",
                "clinical_validity_report": str(cv_path.relative_to(REPO_ROOT)),
                "subgroup_report": str(sg_path.relative_to(REPO_ROOT)) if sg_path.exists() else None,
                "report_generated_on": cv["generated_on"],
            })
    return clinical, subgroup, provenance


Z95 = 1.959964


def holdout_reports_available():
    return all((rep_dir(1 if rep in SYNTHETIC_REPS.get(s, ()) else rep)
                / f"external_validation_{s}_clinical_validity_report.txt").exists()
               for s in SCENARIOS for rep in REPS)


def test_vs_holdout(cohorts):
    """Per split, model and metric: holdout minus test-set estimate, with a 95% interval
    from both sets' bootstrap SEs (the sets hold different patients, so their errors are
    independent): diff +/- 1.96 * sqrt(SE_test^2 + SE_holdout^2), SE = CI width / (2*1.96).

    Until the external_validation_* reports exist, the holdout value is SYNTHETIC: the test
    value plus N(0, SE_holdout), with SE_holdout = SE_test * sqrt(n_test / n_holdout).
    That is pure sampling noise -- no optimism is built in for any model."""
    test, _, _ = load_reports("")
    holdout = load_reports("external_validation_")[0] if holdout_reports_available() else None
    rows = []
    for (scenario, rep), cv in test.items():
        splits = cohorts[scenario]["splits"]
        n_test, n_holdout = splits["test"]["patients"], splits["external_validation"]["patients"]
        for model, m in cv["models"].items():
            rng = random.Random(f"{SYNTHETIC_SEED}-tvh-{scenario}-{rep}-{model}")
            for metric in ("c_index", "brier", "auc"):
                t, lo, hi = m.get(metric), m.get(f"{metric}_lo"), m.get(f"{metric}_hi")
                if None in (t, lo, hi):
                    continue
                se_t = (hi - lo) / (2 * Z95)
                if holdout is not None:
                    hm = holdout[(scenario, rep)]["models"].get(model, {})
                    h, hlo, hhi = hm.get(metric), hm.get(f"{metric}_lo"), hm.get(f"{metric}_hi")
                    if None in (h, hlo, hhi):
                        continue
                    se_h, provenance = (hhi - hlo) / (2 * Z95), "report"
                else:
                    se_h = se_t * math.sqrt(n_test / n_holdout)
                    h = t + rng.gauss(0, se_h)
                    if metric != "brier":
                        h = min(h, RANKING_CAP)
                    provenance = "synthetic"
                diff, se_d = h - t, math.sqrt(se_t ** 2 + se_h ** 2)
                rows.append({"scenario": scenario, "rep": rep, "model": model, "metric": metric,
                             "test": round(t, 4), "holdout": round(h, 4), "difference": round(diff, 4),
                             "ci_low": round(diff - Z95 * se_d, 4), "ci_high": round(diff + Z95 * se_d, 4),
                             "excludes_zero": abs(diff) > Z95 * se_d, "provenance": provenance})
    summary = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["model"], row["metric"])].append(row)
    for (scenario, model, metric), rs in groups.items():
        summary.append({"scenario": scenario, "model": model, "metric": metric, "n_reps": len(rs),
                        "difference_mean": statistics.mean(r["difference"] for r in rs),
                        "reps_excluding_zero": sum(r["excludes_zero"] for r in rs),
                        "provenance": rs[0]["provenance"]})
    return rows, summary


def cohort_summary():
    """Cohort-flow columns for the pooled train+test+holdout patients (identical across reps:
    every rep is a different partition of the same pool), via the repo's own
    cohort_flow_analysis.build_final_cohort_column."""
    os.environ.setdefault("CKD_REP", "1")
    sys.path.insert(0, str(REPO_ROOT))
    import pandas as pd
    from pkgs.data_analysis.cohort_flow_analysis import build_final_cohort_column
    from pkgs.data_analysis.types import ExperimentScenario

    out = {}
    for scenario in SCENARIOS:
        frames = {split: pd.read_csv(rep_dir(1) / f"{scenario}_{split}_data.csv")
                  for split in ("train", "test", "external_validation")}
        pooled = pd.concat(frames.values(), ignore_index=True)
        col = build_final_cohort_column(ExperimentScenario(scenario), pooled)
        per_patient = pooled.groupby("subject_id").agg(rows=("subject_id", "size"),
                                                     event=("has_esrd", "max"))
        col["single_row_fraction"] = float((per_patient["rows"] == 1).mean())
        col["splits"] = {}
        for split, frame in frames.items():
            events = frame.groupby("subject_id")["has_esrd"].max()
            col["splits"][split] = {"patients": int(events.size), "events": int(events.sum()),
                                    "rows": int(len(frame))}
        out[scenario] = col
    return out


def write_csv(path, rows):
    fields = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    out_dir = PAPER_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    clinical, subgroup, provenance = load_reports()
    synthetic = {(p["scenario"], p["rep"]) for p in provenance if p["provenance"] != "report"}

    per_run = []
    for (scenario, rep), cv in clinical.items():
        for model, metrics in cv["models"].items():
            per_run.append({"scenario": scenario, "rep": rep, "model": model, **metrics,
                            "provenance": "synthetic_from_rep1" if (scenario, rep) in synthetic else "report"})

    groups = defaultdict(list)
    for row in per_run:
        groups[(row["scenario"], row["model"])].append(row)
    summary = []
    for (scenario, model), rows in groups.items():
        result = {"scenario": scenario, "model": model, "n_reps": len(rows),
                  "brier_source": rows[0].get("brier_source")}
        for metric in ("c_index", "brier", "auc", "brier_730", "brier_1825"):
            mean, sd = mean_sd([r.get(metric) for r in rows])
            result[f"{metric}_mean"], result[f"{metric}_sd"] = mean, sd
        for metric in ("c_index", "brier", "auc"):
            widths = [(r[f"{metric}_hi"] - r[f"{metric}_lo"]) / 2 for r in rows
                      if r.get(f"{metric}_lo") is not None and r.get(f"{metric}_hi") is not None]
            result[f"{metric}_ci_halfwidth_mean"] = statistics.mean(widths) if widths else None
        summary.append(result)

    paired = []
    for scenario in ("four_features", "eight_features"):
        by_model = defaultdict(list)
        for rep in REPS:
            for model, diff in clinical[(scenario, rep)]["paired_vs_kfre"].items():
                by_model[model].append(diff)
        for model, diffs in by_model.items():
            row = {"scenario": scenario, "model": model, "n_reps": len(diffs)}
            for metric, better in (("c_index", 1), ("brier", -1), ("auc", 1)):
                row[f"{metric}_diff_mean"] = statistics.mean(d[metric] for d in diffs)
                row[f"{metric}_reps_better_excl0"] = sum(
                    d[f"{metric}_excl0"] and d[metric] * better > 0 for d in diffs)
                row[f"{metric}_reps_worse_excl0"] = sum(
                    d[f"{metric}_excl0"] and d[metric] * better < 0 for d in diffs)
            paired.append(row)

    dca = []
    for (scenario, rep), cv in clinical.items():
        for horizon, block in cv["dca"].items():
            for threshold, treat_all in block.get("treat_all", {}).items():
                row = {"scenario": scenario, "rep": rep, "horizon": horizon, "threshold": threshold,
                       "treat_all": treat_all,
                       "egfr_lt30": block.get("eGFR<30", {}).get(threshold),
                       "egfr_lt45": block.get("eGFR<45", {}).get(threshold)}
                for model, curve in block["models"].items():
                    row[model] = curve.get(threshold)
                dca.append(row)

    sub_rows = []
    for (scenario, rep), groups_ in subgroup.items():
        for (dimension, group), info in groups_.items():
            for model, metrics in info["models"].items():
                sub_rows.append({"scenario": scenario, "rep": rep, "dimension": dimension,
                                 "group": group, "n": info["n"], "events": info["events"],
                                 "model": model, **metrics})
    sub_summary = []
    by_key = defaultdict(list)
    for row in sub_rows:
        by_key[(row["scenario"], row["dimension"], row["group"], row["model"])].append(row)
    for (scenario, dimension, group, model), rows in by_key.items():
        c_mean, c_sd = mean_sd([r["c_index"] for r in rows])
        sub_summary.append({"scenario": scenario, "dimension": dimension, "group": group,
                            "model": model, "n_reps": len(rows),
                            "n_mean": statistics.mean(r["n"] for r in rows),
                            "events_mean": statistics.mean(r["events"] for r in rows),
                            "c_index_mean": c_mean, "c_index_sd": c_sd,
                            "brier_mean": mean_sd([r["brier"] for r in rows])[0],
                            "auc_mean": mean_sd([r["auc"] for r in rows])[0]})

    calibration = {f"{s}_rep{r}": cv["dca"] for (s, r), cv in clinical.items() if r == 1}
    cohorts = cohort_summary()
    tvh_rows, tvh_summary = test_vs_holdout(cohorts)

    write_csv(out_dir / "performance_per_run.csv", per_run)
    write_csv(out_dir / "performance_summary.csv", summary)
    write_csv(out_dir / "paired_vs_kfre.csv", paired)
    write_csv(out_dir / "dca_summary.csv", dca)
    write_csv(out_dir / "subgroup_per_run.csv", sub_rows)
    write_csv(out_dir / "subgroup_summary.csv", sub_summary)
    write_csv(out_dir / "test_vs_holdout.csv", tvh_rows)
    write_csv(out_dir / "test_vs_holdout_summary.csv", tvh_summary)
    (out_dir / "calibration_rep1.json").write_text(json.dumps(calibration, indent=1, default=str) + "\n")
    (out_dir / "cohort_summary.json").write_text(json.dumps(cohorts, indent=2, default=str) + "\n")
    (out_dir / "performance_provenance.json").write_text(json.dumps({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).relative_to(REPO_ROOT)),
        "STAND_IN_WARNING": (
            "Holdout (external_validation) numbers are read from the test-set reports, and "
            "twenty_features_heterogeneous reps 2-5 are synthetic (rep1 x U[0.95,1.05]). "
            "Replace both before submission; see this script's docstring."),
        "eval_report_prefix": EVAL_REPORT_PREFIX,
        "synthetic_reps": SYNTHETIC_REPS, "synthetic_seed": SYNTHETIC_SEED,
        "synthetic_range": SYNTHETIC_RANGE,
        "test_vs_holdout": ("report" if holdout_reports_available() else
                            "SYNTHETIC holdout = test + N(0, SE_test*sqrt(n_test/n_holdout)); "
                            "replace once external_validation_* reports exist"),
        "aggregation": "Arithmetic mean and sample SD (n-1) across five patient-level splits.",
        "sources": provenance,
    }, indent=2) + "\n")
    print(f"Wrote {len(per_run)} per-run rows, {len(summary)} summary rows, "
          f"{len(sub_rows)} subgroup rows to {out_dir}")


if __name__ == "__main__":
    main()
