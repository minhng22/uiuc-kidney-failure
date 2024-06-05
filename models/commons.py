"""
----------MIMIC-V data paths----------
"""

diagnose_icd_file_path = '../data/mimic-iv-2.2/hosp/diagnoses_icd.csv'
patients_file_path = '../data/mimic-iv-2.2/hosp/patients.csv'

# Codes extracted from d_icd_diagnoses.csv
def get_kidney_failure_codes():
    return icd_9_kidney_failure_code_g1 + icd_9_kidney_failure_code_g2 + icd_9_kidney_failure_code_g3 + \
             icd_10_kidney_failure_code_g1 + icd_10_kidney_failure_code_g2 + icd_10_kidney_failure_code_g3

icd_10_kidney_failure_code_g1 = ['N17', 'N170', 'N171', 'N172', 'N178', 'N179'] # Desc: Acute kidney failure with ...
icd_10_kidney_failure_code_g2 = ['N19'] # Desc: Unspecified kidney failure ...
icd_10_kidney_failure_code_g3 = ['0904'] # Desc: Postpartum acute kidney failure

icd_9_kidney_failure_code_g1 = ['5845', '5846', '5847', '5848', '5849'] # Desc: Acute kidney failure with ...
icd_9_kidney_failure_code_g2 = ['6393'] # Desc: Kidney failure following abortion and ectopic and molar pregnancies
icd_9_kidney_failure_code_g3 = ['66930', '66932', '66934'] # Desc: Acute kidney failure following labor and delivery ...

"""
----------Generated data paths----------
"""
figs_path = '../generated_data/figs'
figs_path_gender_statistics = figs_path + '/gender_statistics.jpg'
figs_path_age_statistics = figs_path + '/age_statistics.jpg'
figs_path_race_statistics = figs_path + '/race_statistics.jpg'
figs_path_race_stats = figs_path + '/race_statistics.csv'

"""
----------Others----------
"""

# maps to age groups: Children, Young Adults, Middle-aged Adults, Older Adults
# E.g. https://www.semanticscholar.org/paper/Human-Age-Group-Classification-Using-Facial-Bhat-V.K.Patil/19ddb412336ce633c1fe21544605c7bd65ff8d66
age_bins = [0, 16, 30, 45, float('inf')]