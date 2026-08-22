"""
Stage 2 (rep99 mini-experiment) scoped driver.

Runs the 5 ML models + kfre for ONLY the 3 new scenarios (four_features,
eight_features, twenty_features_heterogeneous). Bypasses each experiments
module's `if __name__ == '__main__':` block on purpose: those blocks
(pre-existing Stage 1b code) run CKD_FIFTY_FEATURES_HETEROGENEOUS first,
which is out of scope for Stage 2 and, since that scenario's rep99 data
was deliberately not built (its rep1 source is mid a schema migration by
another session), silently falls through to a full raw MIMIC extraction
from scratch (get_time_series_data_ckd_patients) instead of erroring —
an expensive, unrelated side effect. Calling each module's run_*_model
function directly, scoped to just the 3 new scenarios, avoids that
entirely without editing any previous-stage file.

Usage: CKD_REP=99 PYTHONPATH=. python -m pkgs.scripts.run_stage2_new_scenarios
"""
import sys
import traceback

from pkgs.data_analysis.types import ExperimentScenario
from pkgs.experiments.cox import run_cox_model
from pkgs.experiments.dynamic_deephit import run as run_dynamic_deephit
from pkgs.experiments.hazard_transformer import run as run_hazard_transformer
from pkgs.experiments.logistic_hazard import run as run_logistic_hazard
from pkgs.experiments.rnnsurv import run as run_rnnsurv
from pkgs.experiments.kfre import run_kfre_model

NEW_SCENARIOS = [
    ExperimentScenario.FOUR_FEATURES,
    ExperimentScenario.EIGHT_FEATURES,
    ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS,
]
# kfre is a closed-form equation only published for the 4-/8-variable KFRE.
KFRE_SCENARIOS = [ExperimentScenario.FOUR_FEATURES, ExperimentScenario.EIGHT_FEATURES]

MODELS = [
    ("cox", run_cox_model, NEW_SCENARIOS),
    ("dynamic_deephit", run_dynamic_deephit, NEW_SCENARIOS),
    ("hazard_transformer", run_hazard_transformer, NEW_SCENARIOS),
    ("logistic_hazard", run_logistic_hazard, NEW_SCENARIOS),
    ("rnnsurv", run_rnnsurv, NEW_SCENARIOS),
    ("kfre", run_kfre_model, KFRE_SCENARIOS),
]


def main():
    failed = []
    total = 0
    for name, fn, scenarios in MODELS:
        for scenario in scenarios:
            total += 1
            label = f"{name}/{scenario.value}"
            print(f"==================== Running {label} ====================", flush=True)
            try:
                fn(scenario)
                print(f"✓ {label} completed successfully", flush=True)
            except Exception:
                print(f"✗ {label} failed:", flush=True)
                traceback.print_exc()
                failed.append(label)

    print("=========================================", flush=True)
    print(f"Total: {total}  Failed: {len(failed)}", flush=True)
    if failed:
        print(f"Failed: {failed}", flush=True)
        sys.exit(1)
    print("✓ All scoped Stage 2 experiments completed successfully!", flush=True)


if __name__ == "__main__":
    main()
