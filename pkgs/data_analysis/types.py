from enum import Enum

class ExperimentScenario(Enum):
    NON_TIME_VARIANT = "non_time_variant"
    TIME_VARIANT = "time_variant"
    HETEROGENEOUS = "heterogeneous" # time-variant set up. Use egfr + protein + albumin as features
    HETEROGENEOUS_IMPUTE = "heterogeneous_impute" # like FIVELABMS but impute missing values instead of encoding missingness
    EGFR_COMPONENTS = "egfr_components" # Use gender + age + serum creatinine as features
    FIVELABMS = "fivelabms" # time-variant set up. Use egfr + potassium + urea nitrogen + sodium + chloride as features
    CKD_FIFTY_FEATURES_HETEROGENEOUS = "ckd_fifty_features_heterogeneous" # 50 most common CKD->ESRD lab features
    FOUR_FEATURES = "four_features" # age, gender, egfr, uacr - benchmarked against 4-variable KFRE
    EIGHT_FEATURES = "eight_features" # four_features + calcium, phosphate, bicarbonate, serum_albumin - benchmarked against 8-variable KFRE
    TWENTY_FEATURES_HETEROGENEOUS = "twenty_features_heterogeneous" # top 20 most common CKD->ESRD lab features