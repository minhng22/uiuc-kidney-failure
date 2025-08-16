import pandas as pd
from pkgs.data.store import (get_esrd_patients_and_diagnoses, get_ckd_but_non_esrd_patients_and_diagnoses, 
                             get_lab_events_df_for_patients)
import warnings
warnings.filterwarnings('ignore')

# Define lab codes for analysis
lab_codes_mapping = {
    'creatinine': ['52546', '50912', '52024'],
    'egfr': ['50920', '52026', '53176'], 
    'protein': ['51992', '51102', '51492'],
    'albumin': ['51069', '51070', '52703'],
    'potassium': ['50971'],
    'urea_nitrogen': ['51006', '52647'],
    'sodium': ['50983'],
    'chloride': ['50902'],
    'bicarbonate': ['50882'],
    'anion_gap': ['50868'],
    'hematocrit': ['51221', '52028'],
    'platelet_count': ['51265'],
    'hemoglobin': ['51222', '50811'],
    'glucose': ['50931', '50809', '52027'],
    'calcium_total': ['50893', '52034'],
    'magnesium': ['50960'],
    'phosphate': ['50970'],
    'cholesterol_total': ['50907'],
    'cholesterol_hdl': ['50904'],
    'cholesterol_ldl': ['50905', '50906'],
    'triglycerides': ['51000'],
    'wbc_count': ['51300'],
    'rbc': ['52170'],
    'c_reactive_protein': ['50889'],
    'hemoglobin_a1c': ['50852']
}

def load_combined_data():
    """Load and combine train + test data"""
    print("Loading train and test data...")
    
    train_files = [
        '/home/minhn2/uiuc-kidney-failure/generated_data/rep1/egfr_tv_train_data.csv',
        '/home/minhn2/uiuc-kidney-failure/generated_data/rep1/egfr_tv_test_data.csv'
    ]
    
    combined_data = []
    for file_path in train_files:
        combined_data.append(pd.read_csv(file_path))
    
    return pd.concat(combined_data, ignore_index=True)

def main_analysis():
    """Main analysis function following the specified steps"""
    
    # Step 1: Load combined data
    full_data = load_combined_data()
    if full_data is None:
        return
    
    # Step 2: Print total patients with/without ESRD
    total_patients = full_data['subject_id'].nunique()
    
    print(f"\n=== PATIENT SUMMARY ===")
    print(f"Total patients: {total_patients}")

    esrd_patients = get_esrd_patients_and_diagnoses()
    non_esrd_patients = get_ckd_but_non_esrd_patients_and_diagnoses()
    
    # Step 4: Analyze each lab measurement
    results = []
    
    for lab_name, itemids in lab_codes_mapping.items():
        print(f"\n--- Analyzing {lab_name} ---")
        
        lab_df = get_lab_events_df_for_patients(esrd_patients)
        lab_df_with_items = lab_df[lab_df['itemid'].isin(itemids)]
        
        # check if there are lab records
        
        # Store results
        result = {
            'lab_name': lab_name,
            'total_patients': total_patients,
            'esrd_patients_total': len(esrd_patients),
        }
        
        results.append(result)
    
    # Step 5: Save results to CSV
    results_df = pd.DataFrame(results)
    output_file = '/home/minhn2/uiuc-kidney-failure/lab_timing_analysis_results.csv'
    results_df.to_csv(output_file, index=False)
    
    print(f"\n=== FINAL RESULTS ===")
    print("Results saved to:", output_file)
    print("\nSummary of top labs by coverage:")
    
    if not results_df.empty:
        # Sort by a composite score considering both ESRD and non-ESRD coverage
        results_df['composite_coverage'] = (results_df['esrd_coverage_pct'] + results_df['non_esrd_coverage_pct']) / 2
        top_labs = results_df.sort_values('composite_coverage', ascending=False).head(10)
        
        print(top_labs[['lab_name', 'esrd_coverage_pct', 'non_esrd_coverage_pct', 
                       'condition_x_count', 'condition_y_count', 'condition_both_count']].to_string(index=False))
    
    return results_df

if __name__ == "__main__":
    results = main_analysis()
