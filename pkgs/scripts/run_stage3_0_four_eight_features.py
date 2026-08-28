"""
Stage 3.0 scoped driver: runs only FOUR_FEATURES and EIGHT_FEATURES for a
single model, calling its run function(s) directly instead of going through
the module's `__main__` block (which, for 10 of the 11 models, also
unconditionally trains TWENTY_FEATURES_HETEROGENEOUS).

Per EXPERIMENT_PLAN_DETAILS.md Stage 3.0's scenario-ordering rule:
TWENTY_FEATURES_HETEROGENEOUS must not be launched for any model until the
user has reviewed the four_features/eight_features analysis and separately
approved it. `run_rep.sh` has no per-scenario scope flag and each model's
`__main__` hardcodes all three scenarios, so this driver is the "scoped
driver" mechanism CLAUDE.md's entry-point rule calls for instead of editing
the shared `__main__` blocks.

Usage: python -m pkgs.scripts.run_stage3_0_four_eight_features <model_name>
(invoked once per model by run_stage3_0_four_eight_features.sh, mirroring
run_rep.sh's one-subprocess-per-experiment parallelism)
"""
import sys

from pkgs.data_analysis.types import ExperimentScenario

SCENARIOS = (ExperimentScenario.FOUR_FEATURES, ExperimentScenario.EIGHT_FEATURES)


def run_cox():
    from pkgs.experiments.cox import run_cox_model
    for s in SCENARIOS:
        run_cox_model(s)


def run_dynamic_deephit():
    from pkgs.experiments.dynamic_deephit import run
    for s in SCENARIOS:
        run(s)


def run_hazard_transformer():
    from pkgs.experiments.hazard_transformer import run
    for s in SCENARIOS:
        run(s)


def run_logistic_hazard():
    from pkgs.experiments.logistic_hazard import run
    for s in SCENARIOS:
        run(s)


def run_rnnsurv():
    from pkgs.experiments.rnnsurv import run
    for s in SCENARIOS:
        run(s)


def run_kfre():
    from pkgs.experiments.kfre import run_kfre_model
    for s in SCENARIOS:
        run_kfre_model(s)


def run_deepsurv():
    from pkgs.experiments.deepsurv import run_scenario
    for s in SCENARIOS:
        run_scenario(s)


def run_gbsa():
    from pkgs.experiments.gbsa import run_scenario
    for s in SCENARIOS:
        run_scenario(s)


def run_srf():
    from pkgs.experiments.srf import run_scenario
    for s in SCENARIOS:
        run_scenario(s)


def run_survival_svm():
    from pkgs.experiments.survival_svm import run_scenario
    for s in SCENARIOS:
        run_scenario(s)


def run_weibul():
    from pkgs.experiments.weibul import run_scenario
    for s in SCENARIOS:
        run_scenario(s)


DISPATCH = {
    "cox": run_cox,
    "dynamic_deephit": run_dynamic_deephit,
    "hazard_transformer": run_hazard_transformer,
    "logistic_hazard": run_logistic_hazard,
    "rnnsurv": run_rnnsurv,
    "kfre": run_kfre,
    "deepsurv": run_deepsurv,
    "gbsa": run_gbsa,
    "srf": run_srf,
    "survival_svm": run_survival_svm,
    "weibul": run_weibul,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in DISPATCH:
        print(f"Usage: python -m pkgs.scripts.run_stage3_0_four_eight_features <{'|'.join(DISPATCH)}>")
        sys.exit(1)
    DISPATCH[sys.argv[1]]()
