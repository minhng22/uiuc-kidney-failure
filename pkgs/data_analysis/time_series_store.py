import pandas as pd
from pkgs.commons import diagnose_icd_file_path, ckd_codes_stage3_to_5, esrd_codes
from pkgs.data_analysis.time_series_utils_store import calculate_duration_in_days
from pkgs.data_analysis.types import ExperimentScenario
from pkgs.commons import esrd_codes, patients_file_path, esrd_patient_ids_path
from pkgs.data_analysis.store import (get_egfr_df, get_first_time_esrd_df, get_protein_df, get_albumin_df, 
                                       get_potassium_df, get_urea_nitrogen_df, get_sodium_df, get_chloride_df, 
                                       get_bicarbonate_df, get_anion_gap_df, get_hematocrit_df, get_platelet_count_df, 
                                       get_hemoglobin_df, get_phosphate_df, get_calcium_df, get_glucose_df, 
                                       get_serum_albumin_df, get_wbc_df, get_rbc_df, get_mcv_df, get_mch_df,
                                       get_mchc_df, get_rdw_df, get_magnesium_df, get_uric_acid_df,
                                       get_bilirubin_total_df, get_alt_df, get_ast_df, get_alkaline_phosphatase_df,
                                       get_ldh_df, get_iron_df, get_total_protein_df, get_cholesterol_total_df,
                                       get_triglycerides_df, get_inr_df, get_ptt_df, get_crp_df, get_ferritin_df,
                                       get_transferrin_df, get_tibc_df, get_lactate_df, get_base_excess_df,
                                       get_pco2_df, get_po2_df, get_ph_df, get_bilirubin_direct_df,
                                       get_bilirubin_indirect_df, get_ggt_df, get_amylase_df, get_lipase_df,
                                       get_ck_df, get_troponin_df, get_bnp_df, get_tsh_df, get_free_t4_df,
                                       get_vitamin_d_df, get_pth_df, get_vitamin_b12_df, get_folate_df,
                                       get_reticulocyte_df, get_fibrinogen_df, get_d_dimer_df, get_cortisol_df,
                                       get_hba1c_df, get_ammonia_df, get_osmolality_df, get_lymphocytes_df,
                                       get_neutrophils_df, get_monocytes_df, get_basophils_df, get_eosinophils_df,
                                       get_pt_df, get_rdw_sd_df, get_lab_h_df, get_lab_l_df, get_lab_i_df,
                                       get_urine_specific_gravity_df, get_urine_ph_df, get_uacr_df)
import pandas as pd
from pkgs.data_analysis.types import ExperimentScenario
import numpy as np
import os

# process patients who have progressed to ESRD
def process_positive_patients(diagnoses_df, patient_ids, scenario_name):
    def validate(D):
        filtered_df = D[D['has_esrd'] == 1]
        id_patients_w_lab_records_esrd = filtered_df['subject_id'].unique()

        if len(id_patients_w_lab_records_esrd) != D['subject_id'].nunique():
            print(
                f"Difference"
            )
            for patient in list(set(D['subject_id'].unique()) - set(id_patients_w_lab_records_esrd)):
                print(
                    f"subject_id: {patient} is bad.\n"
                    f"data: {D[D['subject_id'] == patient][['subject_id', 'time', 'has_esrd', 'first_diagnose_esrd_time']]}\n")
    print(
        f"Processing patients who have progressed to ESRD:\n"
        f"Number of patients: {len(patient_ids)}")
    diagnose_esrd_df = diagnoses_df[diagnoses_df['icd_code'].isin(esrd_codes)]
    print(f"Number of patients with ESRD: {diagnose_esrd_df['subject_id'].nunique()}")
    first_time_esrd_df = get_first_time_esrd_df(diagnose_esrd_df)

    patients = pd.read_csv(patients_file_path)
    patients = patients[patients['subject_id'].isin(patient_ids)]

    print(
        f"Stats on patients:\n"
        f"Total patients: {len(patients)}\n"
        f"Dead patients: {patients['dod'].isna().sum()}\n"
        f"Alive patients: {patients['dod'].notna().sum()}"
    )

    lab_df = get_lab_df_for_scenario_name(patients, scenario_name)

    lab_df = pd.merge(lab_df, first_time_esrd_df, on='subject_id', how='left')
    print(
        f'Number of patients after merging: {lab_df["subject_id"].nunique()}\n'
    )

    lab_df['time'] = pd.to_datetime(lab_df['time'])
    lab_df['first_diagnose_esrd_time'] = pd.to_datetime(lab_df['first_diagnose_esrd_time'])

    # Note: This does not work
    # lab_df = lab_df[lab_df['time'] <= lab_df['first_diagnose_esrd_time']]
    # Because a lot of patients don't have lab event records for the admission in which they are diagnosed with ESRD.
    # Example:
    # Patient 11206658: 
    #         subject_id  labevent_id     hadm_id                 time first_diagnose_esrd_time
    # 188809    11206658   14083695.0  21087283.0  2200-02-11 18:20:00      2202-10-11 19:07:00
    # 188810    11206658   14083771.0  21087283.0  2200-02-12 08:07:00      2202-10-11 19:07:00
    # 188811    11206658   14083785.0  21087283.0  2200-02-14 07:45:00      2202-10-11 19:07:00
    # 188812    11206658   14083799.0  21087283.0  2200-02-16 07:00:00      2202-10-11 19:07:00
    # 188813    11206658   14083836.0  21087283.0  2200-02-17 07:55:00      2202-10-11 19:07:00
    # 188814    11206658   14083873.0         NaN  2202-10-11 16:40:00      2202-10-11 19:07:00

    # Initially set has_esrd to 0 for all records
    lab_df['has_esrd'] = 0
    
    lab_df = calculate_duration_in_days(lab_df)

    # empty value means they only have one record.
    lab_df = lab_df.groupby('subject_id').filter(lambda x: x['duration_in_days'].notna().all())

    # Sort by subject_id and time to process records chronologically
    lab_df = lab_df.sort_values(by=['subject_id', 'time'])
    
    # For each patient, find the date of first ESRD diagnosis and mark all records on that day
    def process_patient_esrd(g):
        if g['first_diagnose_esrd_time'].isna().iloc[0]:
            raise ValueError(f"Patient {g['subject_id'].iloc[0]} has no ESRD diagnosis time.")
            
        # Get the date (without time) of the first ESRD diagnosis
        first_esrd_date = g['first_diagnose_esrd_time'].iloc[0].date()
        
        # Mark all records on the same date as has_esrd = 1
        g['has_esrd'] = (g['time'].dt.date == first_esrd_date).astype(int)
        
        # Filter out records after the first ESRD date
        return g[g['time'].dt.date <= first_esrd_date]
    
    # Apply the processing to each patient
    lab_df = lab_df.groupby('subject_id', group_keys=False).apply(process_patient_esrd)
    
    # Keep only patients who have at least one ESRD record
    progressed_patients_ids = lab_df.loc[lab_df['has_esrd'] == 1, 'subject_id'].unique()
    lab_df = lab_df[lab_df['subject_id'].isin(progressed_patients_ids)]
    print(f"Number of ESRD who have lab records at or prior to their esrd diagnose: {len(progressed_patients_ids)}")
    print(
        f"Number of patients after filtering out records after the first 'has_esrd' == 1: {lab_df['subject_id'].nunique()}\n"
        f"Number of records: {len(lab_df)}. Records sample:\n{lab_df[['subject_id', 'time', 'has_esrd', 'first_diagnose_esrd_time']].head()}")

    validate(lab_df)

    print(
        f"Stats on eGFR:\n"
        f"Number of records: {len(lab_df)}. Number of patients: {lab_df['subject_id'].nunique()}\n"
        f"mean {lab_df['egfr'].mean():.3f} sd {lab_df['egfr'].std():.3f}")

    return lab_df


# process patients who have not progressed to ESRD
def process_negative_patients(patient_ids: any, scenario_name: ExperimentScenario):
    print(
        f"Processing patients who have not progressed to ESRD:\n"
        f"Number of patients: {len(patient_ids)}")

    patients = pd.read_csv(patients_file_path)
    patients = patients[patients['subject_id'].isin(patient_ids)]

    print(
        f"Stats on patients:\n"
        f"Total patients: {len(patients)}\n"
        f"Dead patients: {patients['dod'].isna().sum()}\n"
        f"Alive patients: {patients['dod'].notna().sum()}"
    )
    lab_df = get_lab_df_for_scenario_name(patients, scenario_name)

    lab_df['has_esrd'] = 0

    lab_df['time'] = pd.to_datetime(lab_df['time'])
    
    lab_df = calculate_duration_in_days(lab_df)

    # drop subject where there are missing values in duration_in_days
    # In mimic-iv, these are:
    #            has_esrd  duration_in_days
    # 54828      False               NaN
    # 61737      False               NaN
    # 92289      False               NaN
    # 163716     False               NaN
    # 223727     False               NaN
    # 324347     False               NaN
    # 447693     False               NaN
    # These subject have not progressed to ESRD, and only has one record.
    lab_df = lab_df.groupby('subject_id').filter(lambda x: x['duration_in_days'].notna().all())
    print(
        f"Number of patients after filtering out NaN duration_in_days: {lab_df['subject_id'].nunique()}\n"
        f"Records sample:\n{lab_df[['subject_id', 'time', 'has_esrd']].head()}")

    print(
        f"Stats on eGFR:\n"
        f"Number of records: {len(lab_df)}. Number of patients: {lab_df['subject_id'].nunique()}\n"
        f"mean {lab_df['egfr'].mean():.3f} sd {lab_df['egfr'].std():.3f}")
    
    return lab_df

def merge_nearest_within_admission(anchor_df, other_df, value_col, tolerance=None, by='hadm_id'):
    """Attach `value_col` from other_df onto anchor_df, matching each anchor row to the other_df
    row with the nearest charttime, grouped by `by` (default: 'hadm_id', i.e. bounded to the same
    hospital admission). Pass by='subject_id' to match across a patient's whole history instead -
    used for uACR after the 1c-0 pilot showed the same-admission bound caused severe,
    outcome-correlated attrition (96.5% of patients dropped, disproportionately ESRD-negative ones
    - see EXPERIMENT_PLAN_DETAILS.md "1c-0" pilot findings). `tolerance` additionally bounds how
    far apart (in time) a match may be - None means bounded only by the `by` grouping itself (no
    separate time-window on top of that).

    Design + literature backing (Tangri et al. 2016 JAMA eAppendix 1's Geisinger cohort precedent;
    APACHE II's 24h-window convention for chemistry-panel labs; landmarking methodology for sparse
    time-varying covariates) is recorded in EXPERIMENT_PLAN_DETAILS.md, section "1a-2"."""
    left = anchor_df.copy()
    left['charttime'] = pd.to_datetime(left['charttime'])
    left = left.sort_values('charttime')

    right = other_df[[by, 'charttime', value_col]].dropna(subset=[by, 'charttime', value_col]).copy()
    right['charttime'] = pd.to_datetime(right['charttime'])
    right = right.sort_values('charttime')

    merged = pd.merge_asof(left, right, on='charttime', by=by, direction='nearest', tolerance=tolerance)
    return merged

def get_lab_df_for_scenario_name(patients: any, scenario_name: ExperimentScenario):
    if scenario_name in [ExperimentScenario.FOUR_FEATURES, ExperimentScenario.EIGHT_FEATURES, ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS]:
        # Source-population marker for pkgs/data_analysis/cohort_flow_analysis.py (see
        # EXPERIMENT_PLAN_DETAILS.md "1c-0") - this is the pre-lab-filtering CKD stage 3-5
        # cohort (one of the ESRD-positive/negative halves; the two calls per rep sum to the total).
        print(f'COHORT_FLOW|{scenario_name.value}|source_population|patients={patients["subject_id"].nunique()}|records={len(patients)}')
    if scenario_name == ExperimentScenario.TIME_VARIANT or scenario_name == ExperimentScenario.NON_TIME_VARIANT:
        lab_df = get_egfr_df(patients)
    elif scenario_name == ExperimentScenario.HETEROGENEOUS:
        egfr_df = get_egfr_df(patients)
        egfr_df['egfr_missing'] = 0
        egfr_df['protein_missing'] = 1; egfr_df['protein'] = 0
        egfr_df['albumin_missing'] = 1; egfr_df['albumin'] = 0

        print('number of patients with egfr:', egfr_df['subject_id'].nunique())
        print('number of records with egfr:', len(egfr_df))

        protein_df = get_protein_df(patients)
        protein_df['protein_missing'] = 0
        protein_df['egfr_missing'] = 1; protein_df['egfr'] = 0
        protein_df['albumin_missing'] = 1; protein_df['albumin'] = 0

        print('number of patients with protein:', protein_df['subject_id'].nunique())
        print('number of records with protein:', len(protein_df))

        albumin_df = get_albumin_df(patients)
        albumin_df['albumin_missing'] = 0
        albumin_df['egfr_missing'] = 1; albumin_df['egfr'] = 0
        albumin_df['protein_missing'] = 1; albumin_df['protein'] = 0

        print('number of patients with albumin:', albumin_df['subject_id'].nunique())
        print('number of records with albumin:', len(albumin_df))
        
        lab_df = pd.concat([egfr_df, protein_df, albumin_df])
    elif scenario_name == ExperimentScenario.FIVELABMS:
        # eGFR + 9 additional lab measurements
        egfr_df = get_egfr_df(patients)
        egfr_df['egfr_missing'] = 0
        for lab in ['potassium', 'urea_nitrogen', 'sodium', 'chloride', 'bicarbonate', 'anion_gap', 'hematocrit', 'platelet_count', 'hemoglobin']:
            egfr_df[f'{lab}_missing'] = 1
            egfr_df[lab] = 0

        print('number of patients with egfr:', egfr_df['subject_id'].nunique())
        print('number of records with egfr:', len(egfr_df))

        # Get all lab measurements
        lab_dfs = [egfr_df]
        lab_functions = [
            # ('urea_nitrogen', get_urea_nitrogen_df),
            # ('sodium', get_sodium_df),
            # ('chloride', get_chloride_df),
            # ('bicarbonate', get_bicarbonate_df),
            # ('anion_gap', get_anion_gap_df),
            # ('hematocrit', get_hematocrit_df),
            # ('platelet_count', get_platelet_count_df),
            ('hemoglobin', get_hemoglobin_df)
        ]

        all_labs = ['egfr', 'hemoglobin']
        
        for lab_name, lab_func in lab_functions:
            lab_df = lab_func(patients)
            lab_df[f'{lab_name}_missing'] = 0
            
            # Set missing indicators for all other labs
            for other_lab in all_labs:
                if other_lab != lab_name:
                    lab_df[f'{other_lab}_missing'] = 1
                    lab_df[other_lab] = 0

            print(f'number of patients with {lab_name}:', lab_df['subject_id'].nunique())
            print(f'number of records with {lab_name}:', len(lab_df))
            lab_dfs.append(lab_df)
        
        lab_df = pd.concat(lab_dfs)
    elif scenario_name == ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS:
        # 50 lab features aligned to generated_data/rep5/esrd_lab_analysis_report.txt's top 50
        # (by itemid). 12 of the original 50 that weren't in that report (lactate, base_excess,
        # pco2, po2, bilirubin_direct, bilirubin_indirect, ggt, amylase, lipase, ck, troponin,
        # bnp) were swapped for the 12 report items that (a) are in the report's top 50 and
        # (b) actually carry a numeric `valuenum` in labevents.csv. The other 10 report items
        # (Estimated GFR MDRD - itemid 50920 has zero valuenum entries in the whole dataset;
        # Specimen Type/Urine Color/Urine Appearance/Leukocytes[urine]/Bilirubin[urine]/
        # Blood[urine] - text-only dipstick codes like "NEG"/"TR"/"SM", 0% valuenum coverage;
        # Glucose[urine]/Protein[urine] - <28% valuenum coverage, mostly text; Length of Urine
        # Collection - <2% valuenum coverage) could not be added as continuous features without
        # inventing categorical/ordinal encoding not used anywhere else in this codebase, so the
        # 10 lowest-priority (least renal-specific) of the original 50 were kept in their place
        # instead of being dropped. uric_acid, ldh, iron, total_protein, cholesterol_total,
        # triglycerides, crp, ferritin, transferrin, tibc are retained; 28/50 already matched the
        # report to begin with. See CKD_FIFTY_FEATURES_EXPERIMENT_PLAN.md for the full audit.
        all_labs = [
            'egfr', 'urea_nitrogen', 'hemoglobin', 'serum_albumin', 'potassium', 'sodium',
            'bicarbonate', 'phosphate', 'calcium', 'glucose', 'chloride', 'anion_gap',
            'hematocrit', 'platelet_count', 'wbc', 'rbc', 'mcv', 'mch', 'mchc', 'rdw',
            'magnesium', 'uric_acid', 'bilirubin_total', 'alt', 'ast', 'alkaline_phosphatase',
            'ldh', 'iron', 'total_protein', 'cholesterol_total', 'triglycerides', 'inr',
            'ptt', 'crp', 'ferritin', 'transferrin', 'tibc', 'lymphocytes', 'neutrophils',
            'monocytes', 'basophils', 'eosinophils', 'pt', 'rdw_sd', 'lab_h', 'lab_l', 'lab_i',
            'urine_specific_gravity', 'urine_ph', 'ph'
        ]
        
        # Start with eGFR as base - build all columns at once to avoid fragmentation
        egfr_df = get_egfr_df(patients)
        missing_cols = {f'{lab}_missing': (0 if lab == 'egfr' else 1) for lab in all_labs}
        value_cols = {lab: 0 for lab in all_labs if lab != 'egfr'}
        egfr_df = pd.concat([egfr_df, pd.DataFrame({**missing_cols, **value_cols}, index=egfr_df.index)], axis=1)

        print('number of patients with egfr:', egfr_df['subject_id'].nunique())
        print('number of records with egfr:', len(egfr_df))

        lab_dfs = [egfr_df]
        lab_functions = [
            ('urea_nitrogen', get_urea_nitrogen_df), ('hemoglobin', get_hemoglobin_df),
            ('serum_albumin', get_serum_albumin_df), ('potassium', get_potassium_df),
            ('sodium', get_sodium_df), ('bicarbonate', get_bicarbonate_df),
            ('phosphate', get_phosphate_df), ('calcium', get_calcium_df),
            ('glucose', get_glucose_df), ('chloride', get_chloride_df),
            ('anion_gap', get_anion_gap_df), ('hematocrit', get_hematocrit_df),
            ('platelet_count', get_platelet_count_df), ('wbc', get_wbc_df),
            ('rbc', get_rbc_df), ('mcv', get_mcv_df), ('mch', get_mch_df),
            ('mchc', get_mchc_df), ('rdw', get_rdw_df), ('magnesium', get_magnesium_df),
            ('uric_acid', get_uric_acid_df), ('bilirubin_total', get_bilirubin_total_df),
            ('alt', get_alt_df), ('ast', get_ast_df), ('alkaline_phosphatase', get_alkaline_phosphatase_df),
            ('ldh', get_ldh_df), ('iron', get_iron_df), ('total_protein', get_total_protein_df),
            ('cholesterol_total', get_cholesterol_total_df), ('triglycerides', get_triglycerides_df),
            ('inr', get_inr_df), ('ptt', get_ptt_df), ('crp', get_crp_df),
            ('ferritin', get_ferritin_df), ('transferrin', get_transferrin_df),
            ('tibc', get_tibc_df), ('ph', get_ph_df),
            ('lymphocytes', get_lymphocytes_df), ('neutrophils', get_neutrophils_df),
            ('monocytes', get_monocytes_df), ('basophils', get_basophils_df),
            ('eosinophils', get_eosinophils_df), ('pt', get_pt_df), ('rdw_sd', get_rdw_sd_df),
            ('lab_h', get_lab_h_df), ('lab_l', get_lab_l_df), ('lab_i', get_lab_i_df),
            ('urine_specific_gravity', get_urine_specific_gravity_df), ('urine_ph', get_urine_ph_df),
        ]
        
        for lab_name, lab_func in lab_functions:
            lab_df_temp = lab_func(patients)
            # Build all columns at once to avoid fragmentation
            missing_cols = {f'{lab}_missing': (0 if lab == lab_name else 1) for lab in all_labs}
            value_cols = {lab: 0 for lab in all_labs if lab != lab_name}
            lab_df_temp = pd.concat([lab_df_temp, pd.DataFrame({**missing_cols, **value_cols}, index=lab_df_temp.index)], axis=1)

            print(f'number of patients with {lab_name}:', lab_df_temp['subject_id'].nunique())
            print(f'number of records with {lab_name}:', len(lab_df_temp))
            lab_dfs.append(lab_df_temp)
        
        lab_df = pd.concat(lab_dfs)
    elif scenario_name == ExperimentScenario.FOUR_FEATURES or scenario_name == ExperimentScenario.EIGHT_FEATURES:
        # age, gender, egfr, uacr (+ calcium, phosphate, bicarbonate, serum_albumin for
        # EIGHT_FEATURES). No missingness flags - like EGFR_COMPONENTS - real simultaneous values
        # only. Merge design + literature backing (anchor on each creatinine draw; uACR matched
        # nearest-value across the patient's whole history - loosened from same-admission after the
        # 1c-0 pilot showed that bound caused severe, outcome-correlated attrition; chemistry-panel
        # labs matched nearest-value within +/-24h, per APACHE II's same-panel-snapshot convention)
        # recorded in EXPERIMENT_PLAN_DETAILS.md, section "1a-2".
        def log_cohort_flow_stage(stage_name, df):
            # Parseable by pkgs/data_analysis/cohort_flow_analysis.py - see EXPERIMENT_PLAN_DETAILS.md "1c-0".
            print(f'COHORT_FLOW|{scenario_name.value}|{stage_name}|patients={df["subject_id"].nunique()}|records={len(df)}')

        egfr_df = get_egfr_df(patients)
        egfr_df = egfr_df.dropna(subset=['hadm_id'])
        egfr_df['gender'] = egfr_df['gender'].map({'M': 1, 'F': 0})
        log_cohort_flow_stage('has_egfr_admission_linked', egfr_df)

        uacr_df = get_uacr_df(patients)
        lab_df = merge_nearest_within_admission(egfr_df, uacr_df, 'uacr', tolerance=None, by='subject_id')
        lab_df = lab_df.dropna(subset=['uacr'])
        log_cohort_flow_stage('has_qualifying_uacr', lab_df)

        if scenario_name == ExperimentScenario.EIGHT_FEATURES:
            chem_panel_window = pd.Timedelta(hours=24)
            for lab_name, lab_func in [
                ('calcium', get_calcium_df), ('phosphate', get_phosphate_df),
                ('bicarbonate', get_bicarbonate_df), ('serum_albumin', get_serum_albumin_df),
            ]:
                lab_df = merge_nearest_within_admission(lab_df, lab_func(patients), lab_name, tolerance=chem_panel_window)
                lab_df = lab_df.dropna(subset=[lab_name])
                log_cohort_flow_stage(f'has_qualifying_{lab_name}', lab_df)
    elif scenario_name == ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS:
        # Top 20 most common CKD->ESRD lab features, by frequency, per Task A's confirmed ranking
        # (generated_data/rep1/twenty_features_lab_analysis_report.txt). Same heterogeneous
        # missingness-flag pattern as CKD_FIFTY_FEATURES_HETEROGENEOUS, just a 20-item subset; item
        # #1 (Creatinine) is represented as 'egfr' (computed via get_egfr_df), matching that
        # scenario's convention, rather than raw serum_creatinine.
        all_labs = [
            'egfr', 'potassium', 'urea_nitrogen', 'sodium', 'chloride', 'bicarbonate', 'anion_gap',
            'hematocrit', 'platelet_count', 'hemoglobin', 'wbc', 'mchc', 'mch', 'rbc', 'mcv', 'rdw',
            'glucose', 'calcium', 'magnesium', 'phosphate'
        ]

        egfr_df = get_egfr_df(patients)
        missing_cols = {f'{lab}_missing': (0 if lab == 'egfr' else 1) for lab in all_labs}
        value_cols = {lab: 0 for lab in all_labs if lab != 'egfr'}
        egfr_df = pd.concat([egfr_df, pd.DataFrame({**missing_cols, **value_cols}, index=egfr_df.index)], axis=1)

        print('number of patients with egfr:', egfr_df['subject_id'].nunique())
        print('number of records with egfr:', len(egfr_df))

        lab_dfs = [egfr_df]
        lab_functions = [
            ('potassium', get_potassium_df), ('urea_nitrogen', get_urea_nitrogen_df),
            ('sodium', get_sodium_df), ('chloride', get_chloride_df),
            ('bicarbonate', get_bicarbonate_df), ('anion_gap', get_anion_gap_df),
            ('hematocrit', get_hematocrit_df), ('platelet_count', get_platelet_count_df),
            ('hemoglobin', get_hemoglobin_df), ('wbc', get_wbc_df), ('mchc', get_mchc_df),
            ('mch', get_mch_df), ('rbc', get_rbc_df), ('mcv', get_mcv_df), ('rdw', get_rdw_df),
            ('glucose', get_glucose_df), ('calcium', get_calcium_df), ('magnesium', get_magnesium_df),
            ('phosphate', get_phosphate_df),
        ]

        for lab_name, lab_func in lab_functions:
            lab_df_temp = lab_func(patients)
            missing_cols = {f'{lab}_missing': (0 if lab == lab_name else 1) for lab in all_labs}
            value_cols = {lab: 0 for lab in all_labs if lab != lab_name}
            lab_df_temp = pd.concat([lab_df_temp, pd.DataFrame({**missing_cols, **value_cols}, index=lab_df_temp.index)], axis=1)

            print(f'number of patients with {lab_name}:', lab_df_temp['subject_id'].nunique())
            print(f'number of records with {lab_name}:', len(lab_df_temp))
            lab_dfs.append(lab_df_temp)

        lab_df = pd.concat(lab_dfs)
        print(f'COHORT_FLOW|{scenario_name.value}|has_any_of_20_labs|patients={lab_df["subject_id"].nunique()}|records={len(lab_df)}')
    else:
        assert scenario_name == ExperimentScenario.EGFR_COMPONENTS, f"Unknown scenario name: {scenario_name}"
        lab_df = get_egfr_df(patients)
        lab_df['gender'] = lab_df['gender'].map({'M': 1, 'F': 0})
    
    lab_df.rename(columns={'anchor_age': 'age', 'charttime': 'time'}, inplace=True)

    lab_df[get_feature_columns(scenario_name)] = lab_df[get_feature_columns(scenario_name)].replace('', np.nan)
    lab_df.dropna(subset=get_feature_columns(scenario_name), inplace=True)

    print('Finished getting raw lab records for scenario:', scenario_name)
    print(lab_df.head())
    print(lab_df.columns.tolist())

    return lab_df

def get_feature_columns(scenario):
    if scenario == ExperimentScenario.TIME_VARIANT or scenario == ExperimentScenario.NON_TIME_VARIANT:
        return ['egfr']
    elif scenario == ExperimentScenario.HETEROGENEOUS:
        return ['egfr', 'protein', 'albumin']
    elif scenario == ExperimentScenario.EGFR_COMPONENTS:
        return ['age', 'gender', 'serum_creatinine']
    elif scenario == ExperimentScenario.FIVELABMS:
        return ['egfr', 'hemoglobin']
    elif scenario == ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS:
        return [
            'egfr', 'urea_nitrogen', 'hemoglobin', 'serum_albumin', 'potassium', 'sodium',
            'bicarbonate', 'phosphate', 'calcium', 'glucose', 'chloride', 'anion_gap',
            'hematocrit', 'platelet_count', 'wbc', 'rbc', 'mcv', 'mch', 'mchc', 'rdw',
            'magnesium', 'uric_acid', 'bilirubin_total', 'alt', 'ast', 'alkaline_phosphatase',
            'ldh', 'iron', 'total_protein', 'cholesterol_total', 'triglycerides', 'inr',
            'ptt', 'crp', 'ferritin', 'transferrin', 'tibc', 'lymphocytes', 'neutrophils',
            'monocytes', 'basophils', 'eosinophils', 'pt', 'rdw_sd', 'lab_h', 'lab_l', 'lab_i',
            'urine_specific_gravity', 'urine_ph', 'ph'
        ]
    elif scenario == ExperimentScenario.FOUR_FEATURES:
        return ['age', 'gender', 'egfr', 'uacr']
    elif scenario == ExperimentScenario.EIGHT_FEATURES:
        return ['age', 'gender', 'egfr', 'uacr', 'calcium', 'phosphate', 'bicarbonate', 'serum_albumin']
    elif scenario == ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS:
        return [
            'egfr', 'potassium', 'urea_nitrogen', 'sodium', 'chloride', 'bicarbonate', 'anion_gap',
            'hematocrit', 'platelet_count', 'hemoglobin', 'wbc', 'mchc', 'mch', 'rbc', 'mcv', 'rdw',
            'glucose', 'calcium', 'magnesium', 'phosphate'
        ]

def add_time_variant_support(df):
    df = df.sort_values(by=['subject_id', 'duration_in_days'])

    df['start'] = df['duration_in_days']
    df['stop'] = df.groupby('subject_id')['start'].shift(-1) + 1e-5
    df['stop'] = df['stop'].fillna(df['start'] + 1e-5)

    df.reset_index(drop=True, inplace=True)
    print('add_time_variant_support cols: ', df.columns.tolist())

    # validate start < stop
    invalid_rows = df[df['stop'] < df['start']]
    if not invalid_rows.empty:
        print("Subjects with 'stop' < 'start':")
        print(invalid_rows['subject_id'].tolist())
        raise ValueError("Invalid data: 'stop' < 'start' for some subjects.")
    else:
        print("No subjects found with 'stop' < 'start'.")
    return df


# get late stage ckd patients and info of their progression to esrd.
# only_esrd set to True returns only patients who have progressed to ESRD.
# there are four scenarios:
# 1. 'time_invariant'
# 2. 'time_variant'
# 3. 'heterogeneous': a variation of time_variant where the lab measurements are contains [egfr, proteinuria]
# 4. 'egfr_components': a variation of time_variant where the features are components of egfr [age, sex, serum_creatinine]
def get_time_series_data_ckd_patients(scenario: ExperimentScenario):
    diagnoses_df = pd.read_csv(diagnose_icd_file_path)
    diagnoses_df = diagnoses_df[diagnoses_df['icd_code'].isin(ckd_codes_stage3_to_5 + esrd_codes)]
    diagnoses_df.dropna()

    esrd_patients = diagnoses_df[diagnoses_df['icd_code'].isin(esrd_codes)]['subject_id'].unique()

    non_esrd_patients = diagnoses_df[~diagnoses_df['subject_id'].isin(esrd_patients)]['subject_id'].unique()
    print(f"Sample patients with esrd: {esrd_patients[:10]}")
    print(f'Number of patients progressed from ckd stage 3-5 to esrd are {len(esrd_patients)} '
          f'over {diagnoses_df["subject_id"].nunique()}, '
          f'accounts for {round(100 * len(esrd_patients)/diagnoses_df["subject_id"].nunique(), 3)}%')
    print(f'Number of patients who have not progressed to esrd are {len(non_esrd_patients)} '
          f'over {diagnoses_df["subject_id"].nunique()}, '
          f'accounts for {round(100 * len(non_esrd_patients)/diagnoses_df["subject_id"].nunique(), 3)}%')
    
    lab_df_1 = process_negative_patients(non_esrd_patients, scenario)
    lab_df_2 = process_positive_patients(diagnoses_df, esrd_patients, scenario)

    lab_df = pd.concat([lab_df_1, lab_df_2])
    print(f"After merge:\n"
          f"Total number of patients: {lab_df['subject_id'].nunique()}\n"
          f"Total number of records: {len(lab_df)}")
    
    lab_df['subject_id'] = lab_df['subject_id'].astype(int)
    lab_df['duration_in_days'] = lab_df['duration_in_days'].astype(float)

    if scenario == ExperimentScenario.NON_TIME_VARIANT:
        # right-censoring. similar to work done by:
        # 1. Hagar et al.: Survival Analysis of EHR CKD Data
        d = pd.DataFrame(columns=lab_df.columns)
        for _, group in lab_df.groupby('subject_id'):
            max_row = group.loc[group['duration_in_days'].idxmax()]
            d = d._append(max_row)
        lab_df = d
    elif scenario == ExperimentScenario.TIME_VARIANT:
        lab_df = add_time_variant_support(lab_df)[['subject_id', 'duration_in_days', 'start', 'stop', 'egfr', 'has_esrd']]
    elif scenario == ExperimentScenario.HETEROGENEOUS:
        lab_df = add_time_variant_support(lab_df)[['subject_id', 'duration_in_days', 'start', 'stop', 'egfr', 'egfr_missing', 'protein', 'protein_missing', 'albumin', 'albumin_missing', 'has_esrd']]
    elif scenario == ExperimentScenario.FIVELABMS:
        lab_df = add_time_variant_support(lab_df)[['subject_id', 'duration_in_days', 'start', 'stop', 'egfr', 'egfr_missing','hemoglobin', 'hemoglobin_missing', 'has_esrd']]
    elif scenario == ExperimentScenario.EGFR_COMPONENTS:
        lab_df = add_time_variant_support(lab_df)[['subject_id', 'duration_in_days', 'start', 'stop', 'age', 'gender', 'serum_creatinine', 'has_esrd']]
    elif scenario == ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS:
        ckd_fifty_cols = ['subject_id', 'duration_in_days', 'start', 'stop', 
                          'egfr', 'egfr_missing', 
                          'urea_nitrogen', 'urea_nitrogen_missing',
                          'hemoglobin', 'hemoglobin_missing',
                          'serum_albumin', 'serum_albumin_missing',
                          'potassium', 'potassium_missing',
                          'sodium', 'sodium_missing',
                          'bicarbonate', 'bicarbonate_missing',
                          'phosphate', 'phosphate_missing',
                          'calcium', 'calcium_missing',
                          'glucose', 'glucose_missing',
                          'chloride', 'chloride_missing',
                          'anion_gap', 'anion_gap_missing',
                          'hematocrit', 'hematocrit_missing',
                          'platelet_count', 'platelet_count_missing',
                          'wbc', 'wbc_missing',
                          'rbc', 'rbc_missing',
                          'mcv', 'mcv_missing',
                          'mch', 'mch_missing',
                          'mchc', 'mchc_missing',
                          'rdw', 'rdw_missing',
                          'magnesium', 'magnesium_missing',
                          'uric_acid', 'uric_acid_missing',
                          'bilirubin_total', 'bilirubin_total_missing',
                          'alt', 'alt_missing',
                          'ast', 'ast_missing',
                          'alkaline_phosphatase', 'alkaline_phosphatase_missing',
                          'ldh', 'ldh_missing',
                          'iron', 'iron_missing',
                          'total_protein', 'total_protein_missing',
                          'cholesterol_total', 'cholesterol_total_missing',
                          'triglycerides', 'triglycerides_missing',
                          'inr', 'inr_missing',
                          'ptt', 'ptt_missing',
                          'crp', 'crp_missing',
                          'ferritin', 'ferritin_missing',
                          'transferrin', 'transferrin_missing',
                          'tibc', 'tibc_missing',
                          'lymphocytes', 'lymphocytes_missing',
                          'neutrophils', 'neutrophils_missing',
                          'monocytes', 'monocytes_missing',
                          'basophils', 'basophils_missing',
                          'eosinophils', 'eosinophils_missing',
                          'pt', 'pt_missing',
                          'rdw_sd', 'rdw_sd_missing',
                          'lab_h', 'lab_h_missing',
                          'lab_l', 'lab_l_missing',
                          'lab_i', 'lab_i_missing',
                          'urine_specific_gravity', 'urine_specific_gravity_missing',
                          'urine_ph', 'urine_ph_missing',
                          'ph', 'ph_missing',
                          'has_esrd']
        lab_df = add_time_variant_support(lab_df)[ckd_fifty_cols]
    elif scenario == ExperimentScenario.FOUR_FEATURES:
        lab_df = add_time_variant_support(lab_df)[['subject_id', 'duration_in_days', 'start', 'stop', 'age', 'gender', 'egfr', 'uacr', 'has_esrd']]
    elif scenario == ExperimentScenario.EIGHT_FEATURES:
        lab_df = add_time_variant_support(lab_df)[['subject_id', 'duration_in_days', 'start', 'stop', 'age', 'gender', 'egfr', 'uacr', 'calcium', 'phosphate', 'bicarbonate', 'serum_albumin', 'has_esrd']]
    elif scenario == ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS:
        twenty_cols = ['subject_id', 'duration_in_days', 'start', 'stop',
                       'egfr', 'egfr_missing',
                       'potassium', 'potassium_missing',
                       'urea_nitrogen', 'urea_nitrogen_missing',
                       'sodium', 'sodium_missing',
                       'chloride', 'chloride_missing',
                       'bicarbonate', 'bicarbonate_missing',
                       'anion_gap', 'anion_gap_missing',
                       'hematocrit', 'hematocrit_missing',
                       'platelet_count', 'platelet_count_missing',
                       'hemoglobin', 'hemoglobin_missing',
                       'wbc', 'wbc_missing',
                       'mchc', 'mchc_missing',
                       'mch', 'mch_missing',
                       'rbc', 'rbc_missing',
                       'mcv', 'mcv_missing',
                       'rdw', 'rdw_missing',
                       'glucose', 'glucose_missing',
                       'calcium', 'calcium_missing',
                       'magnesium', 'magnesium_missing',
                       'phosphate', 'phosphate_missing',
                       'has_esrd']
        lab_df = add_time_variant_support(lab_df)[twenty_cols]

    lab_df.reset_index(drop=True, inplace=True)

    print(f"Data: \n{lab_df.head()}\n"
          f"Total number of patients: {lab_df['subject_id'].nunique()}\n"
          f"Total number of records: {len(lab_df)}")
    return lab_df[get_final_columns(scenario)]

def get_final_columns(scenario):
    if scenario == ExperimentScenario.NON_TIME_VARIANT:
        return ['subject_id', 'duration_in_days', 'egfr', 'has_esrd']
    elif scenario == ExperimentScenario.TIME_VARIANT:
        return ['subject_id', 'duration_in_days', 'start', 'stop', 'egfr', 'has_esrd']
    elif scenario == ExperimentScenario.HETEROGENEOUS:
        return ['subject_id', 'duration_in_days', 'start', 'stop', 'egfr', 'egfr_missing', 'protein', 'protein_missing', 'albumin', 'albumin_missing', 'has_esrd']
    elif scenario == ExperimentScenario.EGFR_COMPONENTS:
        return ['subject_id', 'duration_in_days', 'start', 'stop', 'age', 'gender', 'serum_creatinine', 'has_esrd']
    elif scenario == ExperimentScenario.FIVELABMS:
        return ['subject_id', 'duration_in_days', 'start', 'stop', 'egfr', 'egfr_missing', 'hemoglobin', 'hemoglobin_missing', 'has_esrd']
    elif scenario == ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS:
        return ['subject_id', 'duration_in_days', 'start', 'stop', 
                'egfr', 'egfr_missing', 
                'urea_nitrogen', 'urea_nitrogen_missing',
                'hemoglobin', 'hemoglobin_missing',
                'serum_albumin', 'serum_albumin_missing',
                'potassium', 'potassium_missing',
                'sodium', 'sodium_missing',
                'bicarbonate', 'bicarbonate_missing',
                'phosphate', 'phosphate_missing',
                'calcium', 'calcium_missing',
                'glucose', 'glucose_missing',
                'chloride', 'chloride_missing',
                'anion_gap', 'anion_gap_missing',
                'hematocrit', 'hematocrit_missing',
                'platelet_count', 'platelet_count_missing',
                'wbc', 'wbc_missing',
                'rbc', 'rbc_missing',
                'mcv', 'mcv_missing',
                'mch', 'mch_missing',
                'mchc', 'mchc_missing',
                'rdw', 'rdw_missing',
                'magnesium', 'magnesium_missing',
                'uric_acid', 'uric_acid_missing',
                'bilirubin_total', 'bilirubin_total_missing',
                'alt', 'alt_missing',
                'ast', 'ast_missing',
                'alkaline_phosphatase', 'alkaline_phosphatase_missing',
                'ldh', 'ldh_missing',
                'iron', 'iron_missing',
                'total_protein', 'total_protein_missing',
                'cholesterol_total', 'cholesterol_total_missing',
                'triglycerides', 'triglycerides_missing',
                'inr', 'inr_missing',
                'ptt', 'ptt_missing',
                'crp', 'crp_missing',
                'ferritin', 'ferritin_missing',
                'transferrin', 'transferrin_missing',
                'tibc', 'tibc_missing',
                'lymphocytes', 'lymphocytes_missing',
                'neutrophils', 'neutrophils_missing',
                'monocytes', 'monocytes_missing',
                'basophils', 'basophils_missing',
                'eosinophils', 'eosinophils_missing',
                'pt', 'pt_missing',
                'rdw_sd', 'rdw_sd_missing',
                'lab_h', 'lab_h_missing',
                'lab_l', 'lab_l_missing',
                'lab_i', 'lab_i_missing',
                'urine_specific_gravity', 'urine_specific_gravity_missing',
                'urine_ph', 'urine_ph_missing',
                'ph', 'ph_missing',
                'has_esrd']
    elif scenario == ExperimentScenario.FOUR_FEATURES:
        return ['subject_id', 'duration_in_days', 'start', 'stop', 'age', 'gender', 'egfr', 'uacr', 'has_esrd']
    elif scenario == ExperimentScenario.EIGHT_FEATURES:
        return ['subject_id', 'duration_in_days', 'start', 'stop', 'age', 'gender', 'egfr', 'uacr', 'calcium', 'phosphate', 'bicarbonate', 'serum_albumin', 'has_esrd']
    elif scenario == ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS:
        return ['subject_id', 'duration_in_days', 'start', 'stop',
                'egfr', 'egfr_missing',
                'potassium', 'potassium_missing',
                'urea_nitrogen', 'urea_nitrogen_missing',
                'sodium', 'sodium_missing',
                'chloride', 'chloride_missing',
                'bicarbonate', 'bicarbonate_missing',
                'anion_gap', 'anion_gap_missing',
                'hematocrit', 'hematocrit_missing',
                'platelet_count', 'platelet_count_missing',
                'hemoglobin', 'hemoglobin_missing',
                'wbc', 'wbc_missing',
                'mchc', 'mchc_missing',
                'mch', 'mch_missing',
                'rbc', 'rbc_missing',
                'mcv', 'mcv_missing',
                'rdw', 'rdw_missing',
                'glucose', 'glucose_missing',
                'calcium', 'calcium_missing',
                'magnesium', 'magnesium_missing',
                'phosphate', 'phosphate_missing',
                'has_esrd']

def get_data_with_null_analyze():
    # get_time_series_data_ckd_patients('egfr_components')
    data = get_time_series_data_ckd_patients(ExperimentScenario.TIME_VARIANT)
    # Select rows that contain NaN values
    rows_with_nan = data[data.isnull().any(axis=1)]
    print("Rows with NaN values:\n%s", rows_with_nan)

def get_esrd_patient_ids():
    diagnoses_df = pd.read_csv(diagnose_icd_file_path)
    diagnoses_df = diagnoses_df[diagnoses_df['icd_code'].isin(ckd_codes_stage3_to_5 + esrd_codes)]
    diagnoses_df.dropna()

    esrd_patients = diagnoses_df[diagnoses_df['icd_code'].isin(esrd_codes)]['subject_id'].unique()
    
    if not os.path.exists(esrd_patient_ids_path):
        print(f"Saving esrd patient ids to {esrd_patient_ids_path}")
        pd.DataFrame(esrd_patients, columns=['subject_id']).to_csv(esrd_patient_ids_path, index=False)

if __name__ == '__main__':
    get_time_series_data_ckd_patients(scenario=ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS)
