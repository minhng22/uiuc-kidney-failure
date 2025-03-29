from enum import Enum

class ExperimentScenario(Enum):
    TIME_INVARIANT = "time_invariant"
    TIME_VARIANT = "time_variant"
    HETEROGENEOUS = "heterogeneous" # time-variant set up. Use egfr + protein + albumin as features
    EGFR_COMPONENTS = "egfr_components" # Use gender + age + serum creatinine as features