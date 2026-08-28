"""
Scoped driver (per CLAUDE.md "check a script's actual entry point" rule) to
retrain ONLY hazard_transformer on rep99, now that its c-index/AUC/Brier
evaluation (and therefore its Optuna selection metric) was fixed to read a
fixed common horizon instead of each patient's own observed time (see
pkgs/experiments/hazard_transformer.py's EVAL_HORIZON_DAYS). No other model
is affected by this fix, so this only retrains hazard_transformer -- cox/ddh/
logistic_hazard/rnnsurv/kfre and the 5 extra models are untouched.

Must run with CKD_REP=99.
"""
import sys
sys.path.append('/home/minhn2/uiuc-kidney-failure')
from pkgs.commons import current_rep
assert current_rep == 99, f"must run with CKD_REP=99, got {current_rep}"

from pkgs.data_analysis.types import ExperimentScenario
from pkgs.experiments.hazard_transformer import run

for scenario in (ExperimentScenario.FOUR_FEATURES, ExperimentScenario.EIGHT_FEATURES,
                 ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS):
    print(f"=== retraining hazard_transformer/{scenario} under fixed-horizon eval ===", flush=True)
    try:
        run(scenario)
    except Exception:
        import traceback
        print(f"non-fatal exception for {scenario} (likely the known post-training AUC edge case)", flush=True)
        traceback.print_exc()

print("DONE", flush=True)
