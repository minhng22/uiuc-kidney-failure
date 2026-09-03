"""Aggregate discrimination metrics (C-index / Brier / AUC) across reps.

Parses the "Discrimination metrics" block that
pkgs/data_analysis/clinical_validity_analysis.py writes into every
generated_data/rep<N>/<scenario>_clinical_validity_report.txt, for
whichever reps (1-5) currently have that file for a given scenario, and
reports mean +/- SD per model -- the format needed for the ML4H 2026
Proceedings-track paper's results tables (paper/to submit 2026).

Usage:
    PYTHONPATH=. python -m pkgs.scripts.aggregate_rep_metrics four_features eight_features twenty_features_heterogeneous

Only reps whose report file actually exists for a given scenario are
included -- this is deliberately tolerant of partial completion (e.g. rep2
having eight_features but not yet twenty_features_heterogeneous), and
prints exactly which reps went into each number so nothing is silently
averaged over a mismatched rep set. Missing/partial reps are reported, not
papered over: a model with fewer reps than the scenario's max gets an
explicit note.
"""
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REP_RANGE = range(1, 6)

LINE_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9 \-]*?):\s*c_index=([\-0-9.]+)\s+brier=([\-0-9.]+)\s+auc=([\-0-9.]+)\s*$"
)


def parse_report(path: Path):
    """Returns {model_name: (c_index, brier, auc)} from one report's
    Discrimination metrics block, or {} if the file doesn't exist yet."""
    if not path.exists():
        return {}
    in_block = False
    out = {}
    for line in path.read_text().splitlines():
        if line.strip().startswith("Discrimination metrics"):
            in_block = True
            continue
        if in_block:
            m = LINE_RE.match(line)
            if m:
                name, c, b, a = m.groups()
                out[name.strip()] = (float(c), float(b), float(a))
            elif line.strip() == "" or line.strip().startswith("==="):
                in_block = False
    return out


def fmt(vals):
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.3f} (n=1)"
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals)
    return f"{mean:.3f}±{sd:.3f} (n={len(vals)})"


def aggregate_scenario(scenario: str):
    per_rep = {}
    for rep in REP_RANGE:
        path = REPO_ROOT / "generated_data" / f"rep{rep}" / f"{scenario}_clinical_validity_report.txt"
        parsed = parse_report(path)
        if parsed:
            per_rep[rep] = parsed

    if not per_rep:
        print(f"\n=== {scenario}: NO reports found for any rep 1-5 ===")
        return

    reps_present = sorted(per_rep.keys())
    print(f"\n=== {scenario} (reps with a report: {reps_present}) ===")

    all_models = []
    for parsed in per_rep.values():
        for name in parsed:
            if name not in all_models:
                all_models.append(name)

    header = f"{'Model':<20} {'C-index':<20} {'Brier':<20} {'AUC':<20} reps used"
    print(header)
    print("-" * len(header))
    for model in all_models:
        c_vals, b_vals, a_vals, reps_used = [], [], [], []
        for rep in reps_present:
            row = per_rep[rep].get(model)
            if row is not None:
                c_vals.append(row[0])
                b_vals.append(row[1])
                a_vals.append(row[2])
                reps_used.append(rep)
        note = "" if len(reps_used) == len(reps_present) else f"  ** only reps {reps_used} **"
        print(f"{model:<20} {fmt(c_vals):<20} {fmt(b_vals):<20} {fmt(a_vals):<20}{note}")


if __name__ == "__main__":
    scenarios = sys.argv[1:] or [
        "four_features",
        "eight_features",
        "twenty_features_heterogeneous",
    ]
    for s in scenarios:
        aggregate_scenario(s)
