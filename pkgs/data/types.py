from enum import Enum

class ExperimentScenario(Enum):
    TIME_INVARIANT = "time_invariant"
    TIME_VARIANT = "time_variant"
    HETEROGENEOUS = "heterogeneous"
    EGFR_COMPONENTS = "egfr_components"