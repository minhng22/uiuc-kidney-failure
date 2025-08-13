import pandas as pd
import numpy as np
from pkgs.commons import (diagnose_icd_file_path, ckd_codes_stage3_to_5, esrd_codes, 
                          lab_events_file_path, patients_file_path)
from pkgs.data.store import get_admission_df
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
import warnings
warnings.filterwarnings('ignore')

# Define lab codes for additional measurements beyond the current ones
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

def get_lab_data_for_itemids(patients_df, itemids, lab_name):
    """Get lab data for specific item IDs"""
    print(f"Getting {lab_name} data for {len(itemids)} item codes...")
    
    # Read lab events in chunks to manage memory
    lab_df_chunks = []
    chunk_size = 100000
    
    for chunk in pd.read_csv(lab_events_file_path, chunksize=chunk_size):
        # Filter for our patients and lab codes
        chunk_filtered = chunk[
            (chunk['subject_id'].isin(patients_df['subject_id'])) &
            (chunk['itemid'].astype(str).isin(itemids)) &
            (chunk['valuenum'].notna())
        ]
        
        if len(chunk_filtered) > 0:
            lab_df_chunks.append(chunk_filtered)
    
    if not lab_df_chunks:
        print(f"No data found for {lab_name}")
        return pd.DataFrame()
    
    lab_df = pd.concat(lab_df_chunks, ignore_index=True)
    
    # Merge with patient data to get demographics
    lab_df = pd.merge(lab_df, patients_df[['subject_id', 'gender', 'anchor_age']], on='subject_id', how='left')
    
    # Rename and clean
    lab_df = lab_df.rename(columns={'valuenum': lab_name, 'charttime': 'time'})
    lab_df['time'] = pd.to_datetime(lab_df['time'])
    
    print(f"Found {len(lab_df)} {lab_name} measurements for {lab_df['subject_id'].nunique()} patients")
    
    return lab_df[['subject_id', 'time', lab_name, 'gender', 'anchor_age']]

def analyze_lab_covariance_with_esrd():
    """Analyze covariance of various lab measurements with ESRD outcome"""
    
    # Get patient cohort
    print("Loading patient cohort...")
    diagnoses_df = pd.read_csv(diagnose_icd_file_path)
    diagnoses_df = diagnoses_df[diagnoses_df['icd_code'].isin(ckd_codes_stage3_to_5 + esrd_codes)]
    
    # Get ESRD patients
    esrd_patients = diagnoses_df[diagnoses_df['icd_code'].isin(esrd_codes)]['subject_id'].unique()
    non_esrd_patients = diagnoses_df[~diagnoses_df['subject_id'].isin(esrd_patients)]['subject_id'].unique()
    
    print(f"ESRD patients: {len(esrd_patients)}")
    print(f"Non-ESRD patients: {len(non_esrd_patients)}")
    
    # Get patient demographics
    patients_df = pd.read_csv(patients_file_path)
    patients_df = patients_df[patients_df['subject_id'].isin(diagnoses_df['subject_id'].unique())]
    
    # Add ESRD status
    patients_df['has_esrd'] = patients_df['subject_id'].isin(esrd_patients).astype(int)
    
    results = []
    
    # Analyze each lab measurement
    for lab_name, itemids in lab_codes_mapping.items():
        print(f"\n--- Analyzing {lab_name} ---")
        
        try:
            lab_df = get_lab_data_for_itemids(patients_df, itemids, lab_name)
            
            if lab_df.empty:
                continue
                
            # Calculate summary statistics per patient
            patient_stats = lab_df.groupby('subject_id').agg({
                lab_name: ['mean', 'std', 'count', 'min', 'max']
            }).round(3)
            
            patient_stats.columns = [f'{lab_name}_{stat}' for stat in ['mean', 'std', 'count', 'min', 'max']]
            patient_stats = patient_stats.reset_index()
            
            # Merge with ESRD status
            patient_stats = pd.merge(patient_stats, patients_df[['subject_id', 'has_esrd']], 
                                   on='subject_id', how='left')
            
            # Calculate statistics for ESRD vs non-ESRD groups
            esrd_group = patient_stats[patient_stats['has_esrd'] == 1]
            non_esrd_group = patient_stats[patient_stats['has_esrd'] == 0]
            
            esrd_patients_with_data = len(esrd_group)
            non_esrd_patients_with_data = len(non_esrd_group)
            
            if esrd_patients_with_data < 10 or non_esrd_patients_with_data < 10:
                print(f"Insufficient data: ESRD={esrd_patients_with_data}, Non-ESRD={non_esrd_patients_with_data}")
                continue
            
            # Calculate correlation
            correlation, p_value = pointbiserialr(patient_stats['has_esrd'], patient_stats[f'{lab_name}_mean'])
            
            # Calculate means for both groups
            esrd_mean = esrd_group[f'{lab_name}_mean'].mean()
            non_esrd_mean = non_esrd_group[f'{lab_name}_mean'].mean()
            
            # Calculate coverage (percentage of patients with this lab)
            total_esrd = len(esrd_patients)
            total_non_esrd = len(non_esrd_patients)
            esrd_coverage = (esrd_patients_with_data / total_esrd) * 100
            non_esrd_coverage = (non_esrd_patients_with_data / total_non_esrd) * 100
            
            result = {
                'lab_name': lab_name,
                'esrd_patients_with_data': esrd_patients_with_data,
                'non_esrd_patients_with_data': non_esrd_patients_with_data,
                'esrd_coverage_pct': round(esrd_coverage, 2),
                'non_esrd_coverage_pct': round(non_esrd_coverage, 2),
                'esrd_mean': round(esrd_mean, 3),
                'non_esrd_mean': round(non_esrd_mean, 3),
                'mean_difference': round(abs(esrd_mean - non_esrd_mean), 3),
                'correlation': round(correlation, 4),
                'p_value': round(p_value, 4),
                'total_measurements': len(lab_df)
            }
            
            results.append(result)
            
            print(f"ESRD coverage: {esrd_coverage:.1f}% ({esrd_patients_with_data}/{total_esrd})")
            print(f"Non-ESRD coverage: {non_esrd_coverage:.1f}% ({non_esrd_patients_with_data}/{total_non_esrd})")
            print(f"ESRD mean: {esrd_mean:.3f}, Non-ESRD mean: {non_esrd_mean:.3f}")
            print(f"Correlation with ESRD: {correlation:.4f} (p={p_value:.4f})")
            
        except Exception as e:
            print(f"Error processing {lab_name}: {e}")
            continue
    
    # Convert results to DataFrame and sort
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("No results found!")
        return
    
    # Sort by a composite score considering coverage and correlation
    results_df['composite_score'] = (
        np.abs(results_df['correlation']) * 
        np.sqrt(results_df['esrd_coverage_pct'] * results_df['non_esrd_coverage_pct'] / 100)
    )
    
    results_df = results_df.sort_values('composite_score', ascending=False)
    
    # Display top 15 results
    print("\n" + "="*100)
    print("TOP 15 LAB MEASUREMENTS BY COVARIANCE WITH ESRD:")
    print("="*100)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    top_15 = results_df.head(15)
    print(top_15.to_string(index=False))
    
    # Save results
    results_df.to_csv('/home/minhn2/uiuc-kidney-failure/lab_covariance_results.csv', index=False)
    print(f"\nFull results saved to lab_covariance_results.csv")
    
    # Create visualization
    plt.figure(figsize=(14, 8))
    
    # Plot 1: Coverage vs Correlation
    plt.subplot(2, 2, 1)
    plt.scatter(results_df['esrd_coverage_pct'], np.abs(results_df['correlation']), 
               alpha=0.7, s=60)
    plt.xlabel('ESRD Coverage (%)')
    plt.ylabel('|Correlation with ESRD|')
    plt.title('Coverage vs Correlation Strength')
    
    # Add labels for top labs
    for idx, row in top_15.head(5).iterrows():
        plt.annotate(row['lab_name'], (row['esrd_coverage_pct'], abs(row['correlation'])),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 2: Composite Score
    plt.subplot(2, 2, 2)
    top_10 = top_15.head(10)
    bars = plt.bar(range(len(top_10)), top_10['composite_score'])
    plt.xticks(range(len(top_10)), top_10['lab_name'], rotation=45, ha='right')
    plt.ylabel('Composite Score')
    plt.title('Top 10 Labs by Composite Score')
    
    # Plot 3: Mean Differences
    plt.subplot(2, 2, 3)
    plt.scatter(results_df['mean_difference'], np.abs(results_df['correlation']), 
               alpha=0.7, s=60)
    plt.xlabel('Mean Difference (|ESRD - Non-ESRD|)')
    plt.ylabel('|Correlation with ESRD|')
    plt.title('Mean Difference vs Correlation')
    
    # Plot 4: Coverage comparison
    plt.subplot(2, 2, 4)
    x = np.arange(len(top_10))
    width = 0.35
    plt.bar(x - width/2, top_10['esrd_coverage_pct'], width, label='ESRD Coverage', alpha=0.8)
    plt.bar(x + width/2, top_10['non_esrd_coverage_pct'], width, label='Non-ESRD Coverage', alpha=0.8)
    plt.xticks(x, top_10['lab_name'], rotation=45, ha='right')
    plt.ylabel('Coverage (%)')
    plt.title('Coverage Comparison - Top 10 Labs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/home/minhn2/uiuc-kidney-failure/lab_covariance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df

if __name__ == "__main__":
    results = analyze_lab_covariance_with_esrd()
