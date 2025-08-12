import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pkgs.commons import fivelabms_train_data_path, fivelabms_test_data_path, generate_data_path_latest_rep
import warnings

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

def analysis_1_feature_availability():
    print("=" * 80)
    print("ANALYSIS 1: FEATURE AVAILABILITY PATTERNS")
    print("=" * 80)
    
    fl_train = pd.read_csv(fivelabms_train_data_path)
    fl_test = pd.read_csv(fivelabms_test_data_path)
    
    def analyze_availability(df, dataset_name):
        print(f"\n{dataset_name} Dataset:")
        print("-" * 40)
        
        total_records = len(df)
        
        # eGFR only (all other lab measurements missing)
        egfr_only = df[(df['egfr_missing'] == 0) & 
                       (df['potassium_missing'] == 1) & 
                       (df['urea_nitrogen_missing'] == 1) &
                       (df['sodium_missing'] == 1) &
                       (df['chloride_missing'] == 1)]
        
        # Individual lab measurements only (when eGFR missing)
        potassium_only = df[(df['egfr_missing'] == 1) & 
                           (df['potassium_missing'] == 0) & 
                           (df['urea_nitrogen_missing'] == 1) &
                           (df['sodium_missing'] == 1) &
                           (df['chloride_missing'] == 1)]
        
        # Urea nitrogen only (when eGFR missing)
        urea_nitrogen_only = df[(df['egfr_missing'] == 1) & 
                               (df['potassium_missing'] == 1) & 
                               (df['urea_nitrogen_missing'] == 0) &
                               (df['sodium_missing'] == 1) &
                               (df['chloride_missing'] == 1)]
        
        # Sodium only (when eGFR missing)
        sodium_only = df[(df['egfr_missing'] == 1) & 
                        (df['potassium_missing'] == 1) & 
                        (df['urea_nitrogen_missing'] == 1) &
                        (df['sodium_missing'] == 0) &
                        (df['chloride_missing'] == 1)]
        
        # Chloride only (when eGFR missing)
        chloride_only = df[(df['egfr_missing'] == 1) & 
                          (df['potassium_missing'] == 1) & 
                          (df['urea_nitrogen_missing'] == 1) &
                          (df['sodium_missing'] == 1) &
                          (df['chloride_missing'] == 0)]

        print(f"Total records: {total_records:,}")
        print(f"eGFR only (all other lab measurements missing): {len(egfr_only):,} ({len(egfr_only)/total_records*100:.2f}%)")
        print(f"Potassium only (eGFR missing): {len(potassium_only):,} ({len(potassium_only)/total_records*100:.2f}%)")
        print(f"Urea nitrogen only (eGFR missing): {len(urea_nitrogen_only):,} ({len(urea_nitrogen_only)/total_records*100:.2f}%)")
        print(f"Sodium only (eGFR missing): {len(sodium_only):,} ({len(sodium_only)/total_records*100:.2f}%)")
        print(f"Chloride only (eGFR missing): {len(chloride_only):,} ({len(chloride_only)/total_records*100:.2f}%)")
        
        accounted_for = len(egfr_only) + len(potassium_only) + len(urea_nitrogen_only) + len(sodium_only) + len(chloride_only)
        print(f"Verification: {accounted_for:,} / {total_records:,} records accounted for")
        
        return {
            'total': total_records,
            'egfr_only': len(egfr_only),
            'potassium_only': len(potassium_only),
            'urea_nitrogen_only': len(urea_nitrogen_only),
            'sodium_only': len(sodium_only),
            'chloride_only': len(chloride_only)
        }
    
    train_stats = analyze_availability(fl_train, "TRAINING")
    test_stats = analyze_availability(fl_test, "TEST")
    
    create_availability_visualization(train_stats, test_stats)
    
    return train_stats, test_stats

def create_availability_visualization(train_stats, test_stats):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    categories = ['eGFR Only', 'Potassium Only', 'Urea Nitrogen Only', 'Sodium Only', 'Chloride Only']
    train_counts = [train_stats['egfr_only'], 
                   train_stats['potassium_only'], train_stats['urea_nitrogen_only'], 
                   train_stats['sodium_only'], train_stats['chloride_only']]
    test_counts = [test_stats['egfr_only'], 
                  test_stats['potassium_only'], test_stats['urea_nitrogen_only'],
                  test_stats['sodium_only'], test_stats['chloride_only']]
    
    train_pcts = [count/train_stats['total']*100 for count in train_counts]
    test_pcts = [count/test_stats['total']*100 for count in test_counts]
    
    colors = ['#2E86C1', '#F39C12', '#E74C3C', '#8E44AD', '#17A2B8']
    bars1 = ax1.bar(categories, train_pcts, color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)
    ax1.set_title('Training Data Feature Availability (FIVELABMS)', fontsize=16, fontweight='bold', color='#2C3E50')
    ax1.set_ylabel('Percentage of Records', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.tick_params(axis='y', labelsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, pct in zip(bars1, train_pcts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    bars2 = ax2.bar(categories, test_pcts, color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)
    ax2.set_title('Test Data Feature Availability (FIVELABMS)', fontsize=16, fontweight='bold', color='#2C3E50')
    ax2.set_ylabel('Percentage of Records', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    ax2.tick_params(axis='y', labelsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, pct in zip(bars2, test_pcts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{generate_data_path_latest_rep}/fivelabms_feature_availability_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nFeature availability visualization saved to: {generate_data_path_latest_rep}/fivelabms_feature_availability_analysis.png")
    plt.close()

def main():
    output_file = f'{generate_data_path_latest_rep}/fivelabms_comprehensive_feature_analysis_report.txt'
    
    with open(output_file, 'w') as f:
        f.write("COMPREHENSIVE FEATURE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("ANALYSIS 1: FEATURE AVAILABILITY PATTERNS\n")
        f.write("=" * 80 + "\n")
        train_stats, test_stats = analysis_1_feature_availability()
        
        f.write("\nTRAINING Dataset:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total records: {train_stats['total']:,}\n")
        f.write(f"eGFR only (other labs missing): {train_stats['egfr_only']:,} ({train_stats['egfr_only']/train_stats['total']*100:.2f}%)\n")
        f.write(f"Potassium only (eGFR missing): {train_stats['potassium_only']:,} ({train_stats['potassium_only']/train_stats['total']*100:.2f}%)\n")
        f.write(f"Urea nitrogen only (eGFR missing): {train_stats['urea_nitrogen_only']:,} ({train_stats['urea_nitrogen_only']/train_stats['total']*100:.2f}%)\n")
        f.write(f"Sodium only (eGFR missing): {train_stats['sodium_only']:,} ({train_stats['sodium_only']/train_stats['total']*100:.2f}%)\n")
        f.write(f"Chloride only (eGFR missing): {train_stats['chloride_only']:,} ({train_stats['chloride_only']/train_stats['total']*100:.2f}%)\n")
        
        f.write("\nTEST Dataset:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total records: {test_stats['total']:,}\n")
        f.write(f"eGFR only (other labs missing): {test_stats['egfr_only']:,} ({test_stats['egfr_only']/test_stats['total']*100:.2f}%)\n")
        f.write(f"Potassium only (eGFR missing): {test_stats['potassium_only']:,} ({test_stats['potassium_only']/test_stats['total']*100:.2f}%)\n")
        f.write(f"Urea nitrogen only (eGFR missing): {test_stats['urea_nitrogen_only']:,} ({test_stats['urea_nitrogen_only']/test_stats['total']*100:.2f}%)\n")
        f.write(f"Sodium only (eGFR missing): {test_stats['sodium_only']:,} ({test_stats['sodium_only']/test_stats['total']*100:.2f}%)\n")
        f.write(f"Chloride only (eGFR missing): {test_stats['chloride_only']:,} ({test_stats['chloride_only']/test_stats['total']*100:.2f}%)\n")
    
    print(f"Comprehensive analysis report saved to: {output_file}")
    print("Visualizations saved to:")
    print(f"  - {generate_data_path_latest_rep}/fivelabms_feature_availability_analysis.png") 

if __name__ == "__main__":
    main()
