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
chart_events_file_path = f"{project_dir()}/data/mimic-iv-2.2/icu/chartevents.csv"

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

diabete_codes = []

lab_codes_creatinine = ['52546', '50912', '52024']
lab_codes_egfr = ['50920', '52026', '53176']
lab_codes_proteins_24hr = ['51068']
lab_codes_proteins = ['51992', '51102', '51492']
lab_codes_albumin = ['51069', '51070', '52703']

ace_inhibitor_drugs = ['Captopril', 'Enalapril', 'Lisinopril', 'Ramipril', 'Perindopril', 'Quinapril', 'Benazepril', 'Trandolapril', 'Moexipril', 'Fosinopril']
"""
----------Generated data paths----------
"""
figs_path = f'{project_dir()}/generated_data/figs'
figs_path_icd_stats = figs_path + '/esrd_icds.jpg'

# trained models - time invariant scenario
egfr_ti_gbsa_model_path = f'{project_dir()}/generated_data/egfr_ti_gbsa_model.pkl'
egfr_ti_deepsurv_model_path = f'{project_dir()}/generated_data/egfr_ti_deepsurv_model.pt'
egfr_ti_cox_model_path = f'{project_dir()}/generated_data/egfr_ti_cox_model.pkl'
egfr_ti_srf_model_path = f'{project_dir()}/generated_data/egfr_ti_srf_model.pkl'
egfr_ti_weibul_model_path = f'{project_dir()}/generated_data/egfr_ti_weibul_model.pkl'

# trained models - time variant scenario
egfr_tv_cox_model_path = f'{project_dir()}/generated_data/egfr_tv_cox_model.pkl'
egfr_tv_dynamic_deep_hit_model_path = f'{project_dir()}/generated_data/egfr_tv_ddh_model.pt'
egfr_tv_hazard_transformer_model_path = f'{project_dir()}/generated_data/egfr_tv_hazard_transformer_model.pt'
egfr_tv_rnn_surv_model_path = f'{project_dir()}/generated_data/egfr_tv_rnn_surv_model.pt'

# trained models - heterogeneous scenario
hg_cox_model_path = f'{project_dir()}/generated_data/hg_cox_model.pkl'
hg_dynamic_deep_hit_model_path = f'{project_dir()}/generated_data/hg_ddh_model.pt'
hg_hazard_transformer_model_path = f'{project_dir()}/generated_data/hg_hazard_transformer_model.pt'
hg_rnn_surv_model_path = f'{project_dir()}/generated_data/hg_rnn_surv_model.pt'

# trained models - egfr components scenario
egfr_components_cox_model_path = f'{project_dir()}/generated_data/egfr_components_cox_model.pkl'
egfr_components_rnn_surv_model_path = f'{project_dir()}/generated_data/egfr_components_rnn_surv_model.pt'
egfr_components_dynamic_deep_hit_model_path = f'{project_dir()}/generated_data/egfr_components_ddh_model.pt'
egfr_components_hazard_transformer_model_path = f'{project_dir()}/generated_data/egfr_components_hazard_transformer_model.pt'

# train and test data
heterogen_train_data_path = f'{project_dir()}/generated_data/heterogen_train_data.csv'
heterogen_test_data_path = f'{project_dir()}/generated_data/heterogen_test_data.csv'
egfr_tv_train_data_path = f'{project_dir()}/generated_data/egfr_tv_train_data.csv'
egfr_tv_test_data_path = f'{project_dir()}/generated_data/egfr_tv_test_data.csv'
egfr_ti_train_data_path = f'{project_dir()}/generated_data/egfr_ti_train_data.csv'
egfr_ti_test_data_path = f'{project_dir()}/generated_data/egfr_ti_test_data.csv'
egfr_components_train_data_path = f'{project_dir()}/generated_data/egfr_components_train_data.csv'
egfr_components_test_data_path = f'{project_dir()}/generated_data/egfr_components_test_data.csv'

"""
----------Others----------
"""

age_bins = [0, 27, 54, 82, float('inf')]
