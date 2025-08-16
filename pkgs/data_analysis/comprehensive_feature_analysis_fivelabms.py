import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pkgs.commons import generate_data_path_latest_rep
import warnings

from pkgs.data_analysis.model_data_store import get_train_test_data
from pkgs.data_analysis.types import ExperimentScenario

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

def analyze_feature_availability():
    print("=" * 80)
    print("ANALYSIS 1: FEATURE AVAILABILITY PATTERNS")
    print("=" * 80)
    
    fl_train, fl_test = get_train_test_data(ExperimentScenario.FIVELABMS)
    
    def analyze_availability(df, dataset_name):
        print(f"\n{dataset_name} Dataset:")
        print("-" * 40)
        
        total_records = len(df)
        
        all_labs = ['egfr', 'hemoglobin']
        
        lab_stats = {}
        print(f"Total records: {total_records:,}")
        
        for lab in all_labs:
            # Count records where this lab is available (not missing)
            lab_available = df[df[f'{lab}_missing'] == 0]
            count = len(lab_available)
            lab_stats[lab] = count
            lab_name = lab.replace('_', ' ').title()
            print(f"{lab_name} available: {count:,} ({count/total_records*100:.2f}%)")
        
        result = {'total': total_records}
        for lab in all_labs:
            result[f'{lab}_available'] = lab_stats[lab]
        
        return result
    
    train_stats = analyze_availability(fl_train, "TRAINING")
    test_stats = analyze_availability(fl_test, "TEST")
    
    create_availability_visualization(train_stats, test_stats)
    
    return train_stats, test_stats

def create_availability_visualization(train_stats, test_stats):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # All 10 lab measurements
    lab_names = ['eGFR', 'Potassium', 'Urea Nitrogen', 'Sodium', 'Chloride', 
                'Bicarbonate', 'Anion Gap', 'Hematocrit', 'Platelet Count', 'Hemoglobin']
    lab_keys = ['egfr_available', 'potassium_available', 'urea_nitrogen_available', 'sodium_available', 'chloride_available',
               'bicarbonate_available', 'anion_gap_available', 'hematocrit_available', 'platelet_count_available', 'hemoglobin_available']
    
    train_counts = [train_stats.get(key, 0) for key in lab_keys]
    test_counts = [test_stats.get(key, 0) for key in lab_keys]
    
    train_pcts = [count/train_stats['total']*100 for count in train_counts]
    test_pcts = [count/test_stats['total']*100 for count in test_counts]
    
    colors = plt.cm.Set3(range(len(lab_names)))
    
    bars1 = ax1.bar(lab_names, train_pcts, color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)
    ax1.set_title('Training Data - Lab Measurement Availability (FIVELABMS)', fontsize=16, fontweight='bold', color='#2C3E50')
    ax1.set_ylabel('Percentage of Records', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.tick_params(axis='y', labelsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, pct in zip(bars1, train_pcts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    bars2 = ax2.bar(lab_names, test_pcts, color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)
    ax2.set_title('Test Data - Lab Measurement Availability (FIVELABMS)', fontsize=16, fontweight='bold', color='#2C3E50')
    ax2.set_ylabel('Percentage of Records', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    ax2.tick_params(axis='y', labelsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, pct in zip(bars2, test_pcts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{generate_data_path_latest_rep}/fivelabms_feature_availability_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nFeature availability visualization saved to: {generate_data_path_latest_rep}/fivelabms_feature_availability_analysis.png")
    plt.close()

def main():
    output_file = f'{generate_data_path_latest_rep}/fivelabms_comprehensive_feature_analysis_report.txt'
    
    with open(output_file, 'w') as f:
        f.write("COMPREHENSIVE FEATURE ANALYSIS REPORT - EXPANDED FIVELABMS\n")
        f.write("=" * 80 + "\n")
        f.write("Scenario: eGFR + 9 additional lab measurements\n")
        f.write("Lab measurements: Potassium, Urea Nitrogen, Sodium, Chloride, Bicarbonate,\n")
        f.write("                 Anion Gap, Hematocrit, Platelet Count, Hemoglobin\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("ANALYSIS 1: FEATURE AVAILABILITY PATTERNS\n")
        f.write("=" * 80 + "\n")
        train_stats, test_stats = analyze_feature_availability()
        
        f.write("\nTRAINING Dataset:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total records: {train_stats['total']:,}\n\n")
        
        # Individual lab availability
        all_labs = ['egfr', 'potassium', 'urea_nitrogen', 'sodium', 'chloride', 
                   'bicarbonate', 'anion_gap', 'hematocrit', 'platelet_count', 'hemoglobin']
        f.write("Lab Measurement Availability:\n")
        for lab in all_labs:
            lab_display = lab.replace('_', ' ').title()
            count = train_stats.get(f'{lab}_available', 0)
            f.write(f"  {lab_display}: {count:,} ({count/train_stats['total']*100:.2f}%)\n")
        
        f.write("\nTEST Dataset:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total records: {test_stats['total']:,}\n\n")
        
        f.write("Lab Measurement Availability:\n")
        for lab in all_labs:
            lab_display = lab.replace('_', ' ').title()
            count = test_stats.get(f'{lab}_available', 0)
            f.write(f"  {lab_display}: {count:,} ({count/test_stats['total']*100:.2f}%)\n")
    
    print(f"Comprehensive analysis report saved to: {output_file}")
    print("Visualizations saved to:")
    print(f"  - {generate_data_path_latest_rep}/fivelabms_feature_availability_analysis.png") 

if __name__ == "__main__":
    main()
