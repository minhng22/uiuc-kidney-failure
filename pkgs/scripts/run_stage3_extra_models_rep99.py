"""
Scoped driver (per CLAUDE.md "check a script's actual entry point" rule) to
train the 5 extra models (deepsurv/gbsa/srf/survival_svm/weibul) added to
four_features/eight_features/twenty_features_heterogeneous, on rep99. Must
run with CKD_REP=99. Per-scenario/per-model isolation (an exception in one
doesn't abort the rest), matching the pattern already used in this repo's
other experiment __main__ blocks.

four_features/{weibul,survival_svm,srf} are already done (from earlier smoke
tests) and are skipped here (their model files already exist and run_scenario
loads-if-exists). four_features/gbsa was mid-run separately when this was
launched, so it's included here too -- harmless if it's already done by the
time this reaches it (loads the saved model instead of retraining).
"""
import sys
sys.path.append('/home/minhn2/uiuc-kidney-failure')
from pkgs.commons import current_rep
assert current_rep == 99, f"must run with CKD_REP=99, got {current_rep}"

from pkgs.data_analysis.types import ExperimentScenario
import pkgs.experiments.deepsurv as deepsurv
import pkgs.experiments.gbsa as gbsa
import pkgs.experiments.srf as srf
import pkgs.experiments.survival_svm as survival_svm
import pkgs.experiments.weibul as weibul

MODELS = [
    ("deepsurv", deepsurv.run_scenario),
    ("gbsa", gbsa.run_scenario),
    ("srf", srf.run_scenario),
    ("survival_svm", survival_svm.run_scenario),
    ("weibul", weibul.run_scenario),
]

SCENARIOS = [
    ExperimentScenario.FOUR_FEATURES,
    ExperimentScenario.EIGHT_FEATURES,
    ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS,
]

for scenario in SCENARIOS:
    for name, fn in MODELS:
        print(f"=== {scenario}/{name} ===", flush=True)
        try:
            fn(scenario)
        except Exception:
            import traceback
            print(f"✗ {scenario}/{name} failed:", flush=True)
            traceback.print_exc()

print("ALL DONE", flush=True)
