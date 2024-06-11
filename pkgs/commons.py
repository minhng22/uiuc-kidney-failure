from pathlib import Path


def project_dir():
    current_script_path = Path(__file__)
    return str(current_script_path.parent.parent)


"""
----------MIMIC-V data paths----------
"""

diagnose_icd_file_path = f"{project_dir()}/data/mimic-iv-2.2/hosp/diagnoses_icd.csv"
patients_file_path = f"{project_dir()}/data/mimic-iv-2.2/hosp/patients.csv"
admissions_file_path = f"{project_dir()}/data/mimic-iv-2.2/hosp/admissions.csv"
lab_events_file_path = f"{project_dir()}/data/mimic-iv-2.2/hosp/labevents.csv"
omr_file_path= f"{project_dir()}/data/mimic-iv-2.2/hosp/omr.csv"
prescription_file_path = f"{project_dir()}/data/mimic-iv-2.2/hosp/prescriptions.csv"

# Codes extracted from d_icd_diagnoses.csv
# ESRD ICD codes
icd_10_esrd_code_g1 = ['N17', 'N170', 'N171', 'N172', 'N178', 'N179']  # Desc: Acute kidney failure with ...
icd_10_esrd_code_g2 = ['N19']  # Desc: Unspecified kidney failure ...
icd_9_esrd_code_g1 = ['5845', '5846', '5847', '5848', '5849']  # Desc: Acute kidney failure with ...

esrd_codes = icd_10_esrd_code_g1 + icd_10_esrd_code_g2 + icd_9_esrd_code_g1 # ESRD

# CKD ICD codes
icd_9_ckd_codes_g1 = ['28521']  # Anemia in chronic kidney disease
icd_9_ckd_codes_g2 = ['40300', '40301', '40310', '40311', '40390', '40391']  # Hypertensive chronic kidney disease ...
icd_9_ckd_codes_g3 = ['40400', '40401', '40402', '40403', '40410', '40411',
                      '40412', '40413', '40490', '40491', '40492',
                      '40493']  # Hypertensive heart and chronic kidney disease ...
icd_9_ckd_codes_g4 = ['5851', '5852', '5853', '5854', '5855', '5859']  # Chronic kidney disease
icd_10_ckd_codes_g1 = ['D631']  # Anemia in chronic kidney disease
icd_10_ckd_codes_g2 = ['E0822', 'E0922', 'E1022', 'E1122', 'E1322']  # ... with diabetic chronic kidney disease
icd_10_ckd_codes_g3 = ['I12', 'I120', 'I129', 'I13', 'I130',
                       'I131', 'I1310', 'I1311', 'I132']  # Hypertensive chronic kidney disease
icd_10_ckd_codes_g4 = ['N18', 'N181', 'N182', 'N183', 'N184', 'N185', 'N189']  # Chronic kidney disease

ckd_codes = (
        icd_9_ckd_codes_g1 + icd_9_ckd_codes_g2 + icd_9_ckd_codes_g3 + icd_9_ckd_codes_g4 +
        icd_10_ckd_codes_g1 + icd_10_ckd_codes_g2 + icd_10_ckd_codes_g3 + icd_10_ckd_codes_g4)

ckd_codes_stage3_to_5 = ['5853', '5854', '5855', 'N183', 'N184', 'N185']
ckd_codes_hypertension = icd_10_ckd_codes_g3 + icd_9_ckd_codes_g3
ckd_codes_diabetes_mellitus = icd_10_ckd_codes_g2

creatinine_lab_codes = ['52546', '50912', '52024']
egfr_lab_codes = ['50920', '52026']
proteins_24hr_lab_codes = ['51068']

ace_inhibitor_drugs = ['Captopril', 'Enalapril', 'Lisinopril', 'Ramipril', 'Perindopril', 'Quinapril', 'Benazepril', 'Trandolapril', 'Moexipril', 'Fosinopril']
"""
----------Generated data paths----------
"""
figs_path = '../generated_data/figs'
figs_path_gender_statistics = figs_path + '/esrd_gender_statistics.jpg'
figs_path_age_statistics = figs_path + '/esrd_age_statistics.jpg'
figs_path_race_statistics = figs_path + '/esrd_race_statistics.jpg'
figs_path_race_stats = figs_path + '/esrd_race_statistics.csv'
figs_path_icd_stats = figs_path + '/esrd_icds.jpg'

"""
----------Others----------
"""

# maps to age groups: Children, Young Adults, Middle-aged Adults, Older Adults
# E.g. https://www.semanticscholar.org/paper/Human-Age-Group-Classification-Using-Facial-Bhat-V.K.Patil/19ddb412336ce633c1fe21544605c7bd65ff8d66
age_bins = [0, 16, 30, 45, float('inf')]
