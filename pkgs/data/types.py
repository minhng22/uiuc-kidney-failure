from enum import Enum

class ExperimentScenario(Enum):
    NON_TIME_VARIANT = "non_time_variant"
    TIME_VARIANT = "time_variant"
    HETEROGENEOUS = "heterogeneous" # time-variant set up. Use egfr + protein + albumin as features
    EGFR_COMPONENTS = "egfr_components" # Use gender + age + serum creatinine as features