"""
Stage 3.0 scoped driver: runs only TWENTY_FEATURES_HETEROGENEOUS for a
single model, once the user has approved that scenario per Stage 3.0's
scenario-ordering rule (see run_stage3_0_four_eight_features.py, which
this mirrors for the four_features/eight_features side).

`kfre` is excluded — it has no published equation for
twenty_features_heterogeneous (see EXPERIMENT_PLAN_DETAILS.md's KFRE
section), so only 10 of the 11 models apply here.

Usage: python -m pkgs.scripts.run_stage3_0_twenty_features <model_name>
(invoked once per model by run_rep_stage3_0_twenty.sh, mirroring
run_rep_stage3_0_four_eight.sh's parallelism)
"""
import sys

from pkgs.data_analysis.types import ExperimentScenario

SCENARIO = ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS


def run_cox():
    from pkgs.experiments.cox import run_cox_model
    run_cox_model(SCENARIO)


def run_dynamic_deephit():
    from pkgs.experiments.dynamic_deephit import run
    run(SCENARIO)


def run_hazard_transformer():
    from pkgs.experiments.hazard_transformer import run
    run(SCENARIO)


def run_logistic_hazard():
    from pkgs.experiments.logistic_hazard import run
    run(SCENARIO)


def run_rnnsurv():
    from pkgs.experiments.rnnsurv import run
    run(SCENARIO)


def run_deepsurv():
    from pkgs.experiments.deepsurv import run_scenario
    run_scenario(SCENARIO)


def run_gbsa():
    from pkgs.experiments.gbsa import run_scenario
    run_scenario(SCENARIO)


def run_srf():
    from pkgs.experiments.srf import run_scenario
    run_scenario(SCENARIO)


def run_survival_svm():
    from pkgs.experiments.survival_svm import run_scenario
    run_scenario(SCENARIO)


def run_weibul():
    from pkgs.experiments.weibul import run_scenario
    run_scenario(SCENARIO)


DISPATCH = {
    "cox": run_cox,
    "dynamic_deephit": run_dynamic_deephit,
    "hazard_transformer": run_hazard_transformer,
    "logistic_hazard": run_logistic_hazard,
    "rnnsurv": run_rnnsurv,
    "deepsurv": run_deepsurv,
    "gbsa": run_gbsa,
    "srf": run_srf,
    "survival_svm": run_survival_svm,
    "weibul": run_weibul,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in DISPATCH:
        print(f"Usage: python -m pkgs.scripts.run_stage3_0_twenty_features <{'|'.join(DISPATCH)}>")
        sys.exit(1)
    DISPATCH[sys.argv[1]]()
