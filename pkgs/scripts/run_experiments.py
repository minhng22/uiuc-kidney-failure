"""Parameterized training and analysis for the 4/8/20-feature experiments.

How to run
----------
From the repository root, activate the project Python environment. Choose
"analyze" for existing trained models or "train" to invoke model training/
evaluation (each model's existing function controls reuse of saved models)::

    # All analyses and gap audits, all three scenarios, reps 1-5:
    python -m pkgs.scripts.run_experiments analyze --reps all

    # Run repetitions concurrently, with separate logs for each rep:
    python -m pkgs.scripts.run_experiments analyze --reps all --parallel-reps

    # One analysis for selected repetitions:
    python -m pkgs.scripts.run_experiments analyze --reps 2 3 --analyses clinical_validity
    python -m pkgs.scripts.run_experiments analyze --reps 2 3 --analyses feature_importance

    # Subgroup performance (age/sex/race), PAPER_GAPS_EXPERIMENT_PLAN.md Gap 12:
    python -m pkgs.scripts.run_experiments analyze --reps 2 3 --analyses subgroup

    # Select a scenario and model subset on the mini-experiment repetition:
    python -m pkgs.scripts.run_experiments analyze --reps 99 --scenarios twenty_features_heterogeneous --models cox srf

    # Training/evaluation scoped to particular models and scenarios:
    python -m pkgs.scripts.run_experiments train --reps 1 --models cox --scenarios four_features eight_features
    python -m pkgs.scripts.run_experiments train --reps 99 --models deepsurv gbsa srf survival_svm weibul

    # Preview commands without executing, or show all CLI options:
    python -m pkgs.scripts.run_experiments analyze --reps all --dry-run
    python -m pkgs.scripts.run_experiments --help

Selection options:
- --reps: one or more positive numbers; "all" means production reps 1-5.
  Select rep99 explicitly. Omitted --reps uses CKD_REP, defaulting to 1.
- --parallel-reps: run all selected repetitions concurrently. Tasks within
  each repetition still run sequentially. Default: sequential repetitions.
- --scenarios: four_features, eight_features, twenty_features_heterogeneous;
  all three are selected by default.
- --models: cox, dynamic_deephit, hazard_transformer, logistic_hazard, rnnsurv,
  kfre, deepsurv, gbsa, srf, survival_svm, weibul; defaults to all applicable
  models. KFRE is excluded from feature importance and from twenty-feature
  training/clinical validity. Use dynamic_deephit/rnnsurv on the CLI, even
  though analysis reports use ddh/rnn_surv internally.
- --analyses: clinical_validity, feature_importance, subgroup,
  outcome_definition, prediction_time. All five run by default for "analyze".
  Explicit selections run only those tasks. This option does not select anything
  for "train". prediction_time includes raw lab timing (a large CSV scan per rep).
- Production data/artifacts/logs use generated_data/rep_1/ through rep_5/;
  rep99 and rep100 retain their existing names.
- Logs default to the repetition directory as rep<N>_<task>_<timestamp>.log.
  --log-dir optionally overrides the directory for all selected reps.
- --dry-run: prints the selected worker commands and log paths without
  executing them or creating files.

Inputs, outputs, and execution:
- Both actions require existing <scenario>_train_data.csv and
  <scenario>_test_data.csv under the repetition directory. Analysis also needs
  trained model artifacts there; missing models are logged and skipped.
- Analysis writes <scenario>_shap_analysis_report.txt,
  <scenario>_all_models_feature_importance.png,
  <scenario>_clinical_validity_report.txt, <scenario>_calibration_plot.png,
  and <scenario>_decision_curve_plot.png for the selected analyses.
  Clinical validity also writes c_index_comparison.png, brier_comparison.png,
  and auc_comparison.png across the selected scenarios/models. Subgroup writes
  <scenario>_subgroup_performance_report.txt and no charts.
  The audits write stage_gap6_outcome_definition_report.txt and
  stage_gap8_prediction_time_audit_report.txt. They need raw diagnosis/demographic/
  lab files and existing exports, but no trained models. Analysis never rebuilds
  datasets or retrains models; the backward uACR fix requires new extraction.
- Bootstrap resample count for clinical_validity/subgroup: CKD_N_BOOTSTRAP
  (default 1000). Lower it for a quick smoke run.
- Reports/charts stay under the repetition directory and are overwritten on
  reruns, including subset runs. Use the full selection for final comparisons.
- The runner stays in the foreground, including with --parallel-reps.
  Each rep/task uses a fresh
  process because commons.py binds paths at import time. --worker is internal.
  Training calls selected run functions directly, bypassing experiment main
  blocks. Missing data fails the scenario instead of starting raw extraction.
- Failed workers do not prevent remaining tasks from running; the runner
  returns a nonzero exit status if any worker fails. Inspect analysis logs
  and reports too: analyzers can skip missing models or unavailable metrics
  without failing the whole worker.

"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import importlib
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import traceback
from pkgs.paths import repetition_directory_name


ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = {
    "four_features": "analyze_four_features",
    "eight_features": "analyze_eight_features",
    "twenty_features_heterogeneous": "analyze_twenty_features",
}
TRAIN_FUNCTIONS = {
    "cox": "run_cox_model",
    "dynamic_deephit": "run",
    "hazard_transformer": "run",
    "logistic_hazard": "run",
    "rnnsurv": "run",
    "kfre": "run_kfre_model",
    "deepsurv": "run_scenario",
    "gbsa": "run_scenario",
    "srf": "run_scenario",
    "survival_svm": "run_scenario",
    "weibul": "run_scenario",
}
AUDIT_MODULES = {
    "outcome_definition": "pkgs.scripts.audit_outcome_definition",
    "prediction_time": "pkgs.scripts.audit_prediction_time",
}
ANALYSES = ("clinical_validity", "feature_importance", "subgroup", *AUDIT_MODULES)
MODEL_ALIASES = {"dynamic_deephit": "ddh", "rnnsurv": "rnn_surv"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("train", "analyze"))
    parser.add_argument("--reps", nargs="+", default=[os.environ.get("CKD_REP", "1")],
                        help="Repetition numbers or 'all' (1-5); defaults to CKD_REP or 1")
    parser.add_argument("--parallel-reps", action="store_true",
                        help="Run selected repetitions concurrently; tasks within each rep remain sequential")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--models", nargs="+", choices=TRAIN_FUNCTIONS,
                        help="Model subset; defaults to all applicable models")
    parser.add_argument("--analyses", nargs="+", choices=ANALYSES,
                        default=list(ANALYSES),
                        help="Analyses to run (default: all five, including raw lab timing)")
    parser.add_argument("--log-dir", type=Path,
                        help="Override the default generated_data/rep<N>/ log directory")
    parser.add_argument("--dry-run", action="store_true", help="Print subprocess commands without running them")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.reps == ["all"]:
        args.reps = list(range(1, 6))
    else:
        try:
            args.reps = list(dict.fromkeys(int(rep) for rep in args.reps))
            if any(rep < 1 for rep in args.reps):
                raise ValueError
        except ValueError:
            parser.error("--reps must be positive integers, or 'all' by itself")
    args.scenarios = list(dict.fromkeys(args.scenarios))
    args.analyses = list(dict.fromkeys(args.analyses))
    if args.models:
        args.models = list(dict.fromkeys(args.models))
    if args.worker and (len(args.reps) != 1 or (args.action == "analyze" and len(args.analyses) != 1)):
        parser.error("A worker requires exactly one repetition and one analysis")
    if args.action == "train" and not args.models:
        args.models = list(TRAIN_FUNCTIONS)
    return args


def run_worker(args):
    # Must precede EVERY project import, including ExperimentScenario.
    os.environ["CKD_REP"] = str(args.reps[0])
    from pkgs.commons import generate_data_path_latest_rep
    from pkgs.data_analysis.types import ExperimentScenario

    output_dir = Path(generate_data_path_latest_rep)
    print(f"rep{args.reps[0]} {args.action}: {', '.join(args.scenarios)}; output={output_dir}", flush=True)
    if args.action == "analyze" and args.analyses[0] in AUDIT_MODULES:
        missing = [str(output_dir / f"{scenario}_{split}_data.csv")
                   for scenario in args.scenarios for split in ("train", "test")
                   if not (output_dir / f"{scenario}_{split}_data.csv").is_file()]
        if missing:
            print(f"FAILED: missing existing data: {', '.join(missing)}", flush=True)
            return 1
        task = args.analyses[0]
        audit_args = ["--scenarios", *args.scenarios]
        if task == "prediction_time":
            audit_args.append("--lab-timing")
        status = importlib.import_module(AUDIT_MODULES[task]).main(audit_args)
        print("COMPLETE" if status == 0 else "FAILED", flush=True)
        return status
    analyzer = None
    if args.action == "analyze":
        if args.analyses == ["clinical_validity"]:
            from pkgs.data_analysis.clinical_validity_analysis import ClinicalValidityAnalyzer
            analyzer = ClinicalValidityAnalyzer(output_dir)
        elif args.analyses == ["subgroup"]:
            from pkgs.data_analysis.subgroup_analysis import SubgroupAnalyzer
            analyzer = SubgroupAnalyzer(output_dir)
        elif args.analyses == ["feature_importance"]:
            from pkgs.data_analysis.feature_importance_analysis import FeatureImportanceAnalyzer
            analyzer = FeatureImportanceAnalyzer("_".join(args.scenarios), output_dir)
        else:
            raise ValueError(f"Unsupported worker analysis: {args.analyses}")
        if args.models:
            selected = {MODEL_ALIASES.get(model, model) for model in args.models}
            analyzer.models = [model for model in analyzer.models if model in selected]
        if not analyzer.models:
            raise ValueError("No applicable models selected (KFRE has no feature-importance analysis)")

    failures = []
    for scenario in args.scenarios:
        # get_train_test_data otherwise falls back to raw-data extraction.
        missing = [str(output_dir / f"{scenario}_{split}_data.csv")
                   for split in ("train", "test")
                   if not (output_dir / f"{scenario}_{split}_data.csv").is_file()]
        if missing:
            print(f"FAILED {scenario}: missing existing data: {', '.join(missing)}", flush=True)
            failures.append(scenario)
            continue
        if analyzer is not None:
            try:
                getattr(analyzer, SCENARIOS[scenario])()
                results = (analyzer.all_results
                           if args.analyses in (["clinical_validity"], ["subgroup"])
                           else analyzer.all_importances)
                if not results.get(scenario):
                    raise RuntimeError(f"No analysis results produced for {scenario}")
            except Exception:
                traceback.print_exc()
                failures.append(scenario)
        else:
            for model in args.models:
                if model == "kfre" and scenario == "twenty_features_heterogeneous":
                    print(f"SKIP {scenario}/kfre: no published equation for this scenario", flush=True)
                    continue
                try:
                    module = importlib.import_module(f"pkgs.experiments.{model}")
                    print(f"RUN {scenario}/{model}", flush=True)
                    getattr(module, TRAIN_FUNCTIONS[model])(ExperimentScenario(scenario))
                except Exception:
                    traceback.print_exc()
                    failures.append(f"{scenario}/{model}")
    if analyzer is not None and args.analyses == ["clinical_validity"] and analyzer.all_results:
        analyzer.create_metrics_comparison_charts()
    print(f"{'FAILED: ' + ', '.join(failures) if failures else 'COMPLETE'}", flush=True)
    return int(bool(failures))


def run_rep(args, rep, stamp):
    """Run one repetition's tasks in order, using isolated worker processes."""
    tasks = args.analyses if args.action == "analyze" else ["train"]
    failures = []
    for task in tasks:
        try:
            command = [sys.executable, "-u", "-m", "pkgs.scripts.run_experiments",
                       args.action, "--reps", str(rep), "--scenarios", *args.scenarios, "--worker"]
            if args.models:
                command.extend(["--models", *args.models])
            if args.action == "analyze":
                command.extend(["--analyses", task])
            print(shlex.join(command), flush=True)
            log_dir = args.log_dir if args.log_dir is not None else ROOT / "generated_data" / repetition_directory_name(rep)
            log = log_dir / f"rep{rep}_{task}_{stamp}.log"
            print(f"Log: {log.resolve()}", flush=True)
            if args.dry_run:
                continue
            env = dict(os.environ, CKD_REP=str(rep), PYTHONUNBUFFERED="1")
            env.setdefault("MPLBACKEND", "Agg")
            log_dir.mkdir(parents=True, exist_ok=True)
            with log.open("w") as stream:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            print(f"rep{rep}/{task}: exit {result.returncode}", flush=True)
            if result.returncode:
                failures.append(f"rep{rep}/{task}")
        except OSError as exc:
            print(f"rep{rep}/{task}: failed to run: {exc}", flush=True)
            failures.append(f"rep{rep}/{task}")
    return failures


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        return run_worker(args)

    print(f"Runner PID {os.getpid()} on {socket.gethostname()}", flush=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    failures = []
    # Keep dry-run output ordered and avoid starting threads or workers.
    if args.parallel_reps and not args.dry_run:
        print(f"Running {len(args.reps)} repetitions in parallel", flush=True)
        with ThreadPoolExecutor(max_workers=len(args.reps)) as executor:
            futures = [executor.submit(run_rep, args, rep, stamp) for rep in args.reps]
            for future in futures:
                failures.extend(future.result())
    else:
        for rep in args.reps:
            failures.extend(run_rep(args, rep, stamp))
    if failures:
        print("Failed tasks: " + ", ".join(failures), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())
