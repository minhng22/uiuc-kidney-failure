#!/usr/bin/env python3
"""Print the manuscript's LaTeX table bodies from results/*.csv (run aggregate_results.py first).

    python "paper/to submit 2026/scripts/latex_tables.py" > /tmp/tables.tex

The PLOS manuscript is a single self-contained .tex, so these rows are pasted into it
rather than \\input; rerun and re-paste whenever results/ changes.
"""
import csv
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODEL_ORDER = ["Cox", "Dynamic DeepHit", "Hazard Transformer", "Logistic Hazard", "RNN-Surv",
               "DeepSurv", "GBSA", "Survival RF", "Survival SVM", "Weibull AFT", "KFRE"]
DISPLAY = {"Dynamic DeepHit": "Dynamic-DeepHit"}


def read(name):
    with (RESULTS / name).open() as handle:
        return list(csv.DictReader(handle))


def f3(x):
    return f"{float(x):.3f}"


def discrimination_rows(scenario):
    rows = {r["model"]: r for r in read("performance_summary.csv") if r["scenario"] == scenario}
    models = [m for m in MODEL_ORDER if m in rows]
    best = {
        "c_index": max(models, key=lambda m: float(rows[m]["c_index_mean"])),
        "brier": min(models, key=lambda m: float(rows[m]["brier_mean"])),
        "auc": max(models, key=lambda m: float(rows[m]["auc_mean"])),
    }
    lines = []
    for m in models:
        r = rows[m]
        cells = []
        for metric in ("c_index", "brier", "auc"):
            cell = f"{f3(r[metric + '_mean'])} ({f3(r[metric + '_sd'])})"
            if metric == "brier" and r["brier_source"] != "native":
                cell += "$^{\\dagger}$"
            if best[metric] == m:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        hw = f"$\\pm${f3(r['c_index_ci_halfwidth_mean'])}"
        lines.append(f"{DISPLAY.get(m, m)} & {cells[0]} & {hw} & {cells[1]} & {cells[2]} \\\\")
    return "\n".join(lines)


def paired_rows():
    rows = read("paired_vs_kfre.csv")
    out = []
    for m in MODEL_ORDER[:-1]:
        cells = []
        for scenario in ("four_features", "eight_features"):
            r = next(x for x in rows if x["scenario"] == scenario and x["model"] == m)
            for metric in ("c_index", "brier", "auc"):
                better, worse = int(r[f"{metric}_reps_better_excl0"]), int(r[f"{metric}_reps_worse_excl0"])
                diff = float(r[f"{metric}_diff_mean"])
                tag = f"{better}/5" + (f", {worse}$\\downarrow$" if worse else "")
                cells.append(f"{diff:+.3f} ({tag})".replace("-", "$-$", 1) if diff < 0 else f"{diff:+.3f} ({tag})")
        out.append(f"{DISPLAY.get(m, m)} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


def subgroup_rows():
    rows = read("subgroup_summary.csv")
    columns = [("four_features", "Dynamic DeepHit"), ("four_features", "KFRE"),
               ("eight_features", "Dynamic DeepHit"), ("eight_features", "KFRE"),
               ("twenty_features_heterogeneous", "Dynamic DeepHit"),
               ("twenty_features_heterogeneous", "Hazard Transformer")]
    groups = [("Age group", "18-45", "Age 18--45"), ("Age group", "46-65", "Age 46--65"),
              ("Age group", "66-85", "Age 66--85"), ("Age group", ">85", "Age $>$85"),
              ("Sex as recorded in patients.csv", "F", "Female"),
              ("Sex as recorded in patients.csv", "M", "Male"),
              ("Race", "WHITE", "White"), ("Race", "BLACK", "Black"),
              ("Race", "HISPANIC/LATINO", "Hispanic/Latino"), ("Race", "ASIAN", "Asian"),
              ("Race", "OTHER", "Other"), ("Race", "UNKNOWN/NOT RECORDED", "Unknown/not recorded")]
    out = []
    for dimension, group, label in groups:
        cells = []
        for scenario, model in columns:
            r = next((x for x in rows if x["scenario"] == scenario and x["dimension"] == dimension
                      and x["group"] == group and x["model"] == model and x["c_index_mean"]), None)
            if r is None:
                cells.append("--")
            else:
                reps = int(r["n_reps"])
                cells.append(f3(r["c_index_mean"]) + ("" if reps == 5 else f"$^{{{reps}}}$"))
        out.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


def consistency_rows():
    """Holdout minus test-set estimate, mean over splits, with the number of splits whose
    95% difference interval excluded zero."""
    rows = read("test_vs_holdout_summary.csv")
    out = []
    for m in MODEL_ORDER:
        cells = []
        for scenario in ("four_features", "eight_features", "twenty_features_heterogeneous"):
            for metric in ("c_index", "brier"):
                r = next((x for x in rows if x["scenario"] == scenario and x["model"] == m
                          and x["metric"] == metric), None)
                if r is None:
                    cells.append("--")
                    continue
                diff = float(r["difference_mean"])
                if round(diff, 3) == 0:
                    diff = 0.0  # no "-0.000"
                cell = f"{diff:+.3f} ({r['reps_excluding_zero']}/{r['n_reps']})"
                cells.append(cell.replace("-", "$-$", 1) if diff < 0 else cell)
        out.append(f"{DISPLAY.get(m, m)} & " + " & ".join(cells) + " \\\\")
    per_run = read("test_vs_holdout.csv")
    excluded = sum(r["excludes_zero"] == "True" for r in per_run)
    out.append(f"% {excluded} of {len(per_run)} model-scenario-split-metric comparisons exclude zero; "
               f"provenance={sorted({r['provenance'] for r in per_run})}")
    return "\n".join(out)


def cohort_rows():
    c = json.loads((RESULTS / "cohort_summary.json").read_text())
    days = 365.25

    def col(s):
        x = c[s]
        fu = x["followup"]
        egfr = x["egfr"]
        uacr = x["uacr"]
        sp = x["splits"]
        return {
            "Patients": f"{x['n']:,}",
            "Records (rows)": f"{x['records']:,}",
            "\\% male": f"{x['male_pct']:.1f}",
            "Mean age, yrs (SD)": f"{x['age']['mean']:.1f} ({x['age']['sd']:.1f})",
            "Median eGFR (IQR)": f"{egfr['median']:.1f} ({egfr['q1']:.1f}--{egfr['q3']:.1f})",
            "Median uACR, mg/g (IQR)": "N/A" if uacr is None else
            f"{uacr['median']:.1f} ({uacr['q1']:.1f}--{uacr['q3']:.1f})",
            "Median follow-up, days (IQR)": f"{fu['median'] * days:.1f} ({fu['q1'] * days:.1f}--{fu['q3'] * days:.1f})",
            "ESRD events, patients (\\%)": f"{x['n_events']:,} ({100 * x['n_events'] / x['n']:.1f})",
            "ESRD rate /1,000 person-yrs": f"{x['incidence_rate']['rate']:,.1f}",
            "Train / test / holdout patients": " / ".join(f"{sp[k]['patients']:,}" for k in
                                                          ("train", "test", "external_validation")),
            "Holdout events (\\%)": f"{sp['external_validation']['events']:,} "
            f"({100 * sp['external_validation']['events'] / sp['external_validation']['patients']:.1f})",
        }

    source = {"Patients": "34,332", "Records (rows)": "74,197", "\\% male": "57.2",
              "Mean age, yrs (SD)": "67.8 (16.1)", "ESRD events, patients (\\%)": "31,542 (91.9)"}
    cols = [col(s) for s in ("four_features", "eight_features", "twenty_features_heterogeneous")]
    out = []
    for key in cols[0]:
        out.append(f"{key} & {source.get(key, 'N/A' if key not in ('Train / test / holdout patients', 'Holdout events (\\\\%)') else '---')} & "
                   + " & ".join(c_[key] for c_ in cols) + " \\\\")
    return "\n".join(out)


if __name__ == "__main__":
    for title, body in (("COHORT", cohort_rows()),
                        ("FOUR", discrimination_rows("four_features")),
                        ("EIGHT", discrimination_rows("eight_features")),
                        ("TWENTY", discrimination_rows("twenty_features_heterogeneous")),
                        ("PAIRED", paired_rows()), ("SUBGROUP", subgroup_rows()),
                        ("CONSISTENCY", consistency_rows())):
        print(f"% ---- {title} ----\n{body}\n")
