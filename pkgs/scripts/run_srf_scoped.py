"""
Scoped driver (per CLAUDE.md "check a script's actual entry point" rule) to
run ONLY srf's four_features/eight_features/twenty_features_heterogeneous
scenarios, deliberately skipping run_survival_rf() (the legacy NON_TIME_VARIANT/
egfr_ti scenario).

Why this exists: srf.py's __main__ block calls run_survival_rf() first,
unconditionally, before the three in-scope Stage 3.1 scenarios. That scenario
is not part of Stage 3.1's scope (four/eight/twenty_features_heterogeneous
only, per EXPERIMENT_PLAN_DETAILS.md) -- but because run_survival_rf() runs
synchronously first and its GridSearchCV kept crashing (27+ combined attempts
across rep2/rep3, first from OOM, then from a hardcoded-path bug in
pkgs/commons.py), the actually-in-scope scenarios never even started. Rather
than keep invoking (or debugging) the out-of-scope step, this driver bypasses
it entirely, matching the existing pattern in run_stage3_extra_models_rep99.py.

Run with: CKD_REP=<N> PYTHONPATH=. python -m pkgs.scripts.run_srf_scoped
"""
from pkgs.commons import current_rep
from pkgs.data_analysis.types import ExperimentScenario
import pkgs.experiments.srf as srf

SCENARIOS = [
    ExperimentScenario.FOUR_FEATURES,
    ExperimentScenario.EIGHT_FEATURES,
    ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS,
]

for scenario in SCENARIOS:
    print(f"=== rep{current_rep} srf/{scenario} ===", flush=True)
    try:
        srf.run_scenario(scenario)
    except Exception:
        import traceback
        print(f"✗ rep{current_rep} srf/{scenario} failed:", flush=True)
        traceback.print_exc()

print(f"✓ srf (scoped: four/eight/twenty_features_heterogeneous only) completed for rep{current_rep}", flush=True)
