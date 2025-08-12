import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pkgs.data.store import get_ckd_patients_and_diagnoses
from pkgs.commons import generate_data_path_latest_rep

plt.style.use('default')
sns.set_palette("husl")

def load_data():
    print("Loading ESRD patients data...")
    
    patients_df, esrd_diagnose_df = get_ckd_patients_and_diagnoses(
        late_stage=True,
    )
    
    print("Loading lab items dictionary...")
    d_labitems_path = "/home/minhn2/uiuc-kidney-failure/data/mimic-iv-2.2/hosp/d_labitems.csv"
    d_labitems_df = pd.read_csv(d_labitems_path)
    
    print("Loading lab events data...")
    labevents_path = "/home/minhn2/uiuc-kidney-failure/data/mimic-iv-2.2/hosp/labevents.csv"
    labevents_df = pd.read_csv(labevents_path)
    
    return patients_df, esrd_diagnose_df, d_labitems_df, labevents_df

def analyze_top_lab_measurements(patients_df, labevents_df, d_labitems_df, top_n=10):
    
    print(f"Filtering lab events for {len(patients_df)} ESRD patients...")
    esrd_lab_events = labevents_df[labevents_df['subject_id'].isin(patients_df['subject_id'])]
    
    print(f"Total lab events for ESRD patients: {len(esrd_lab_events):,}")
    print(f"Unique lab measurements (itemids): {esrd_lab_events['itemid'].nunique():,}")
    
    lab_frequency = esrd_lab_events['itemid'].value_counts()
    top_labs = lab_frequency.head(top_n)
    
    lab_descriptions = {}
    for itemid in top_labs.index:
        description = d_labitems_df[d_labitems_df['itemid'] == itemid]['label'].values
        if len(description) > 0:
            lab_descriptions[itemid] = description[0]
        else:
            lab_descriptions[itemid] = f"Unknown (ID: {itemid})"
    
    results = []
    for itemid, count in top_labs.items():
        percentage = (count / len(esrd_lab_events)) * 100
        
        patients_with_test = esrd_lab_events[esrd_lab_events['itemid'] == itemid]['subject_id'].nunique()
        patient_percentage = (patients_with_test / len(patients_df)) * 100
        
        results.append({
            'itemid': itemid,
            'lab_name': lab_descriptions[itemid],
            'total_measurements': count,
            'percentage_of_lab_events': percentage,
            'patients_with_test': patients_with_test,
            'percentage_of_esrd_patients': patient_percentage
        })
    
    results_df = pd.DataFrame(results)
    return results_df, esrd_lab_events

def create_visualizations(results_df, output_dir):
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(results_df)), results_df['total_measurements'], color='skyblue')
    plt.title('Top 10 Most Common Lab Measurements for ESRD Patients', fontsize=14, fontweight='bold')
    plt.xlabel('Lab Measurements', fontsize=12)
    plt.ylabel('Number of Measurements', fontsize=12)
    plt.xticks(range(len(results_df)), [f"{row['itemid']}" for _, row in results_df.iterrows()], rotation=45)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    plt.subplot(2, 2, 2)
    bars = plt.bar(range(len(results_df)), results_df['percentage_of_esrd_patients'], color='lightcoral')
    plt.title('Percentage of ESRD Patients with Each Lab Test', fontsize=14, fontweight='bold')
    plt.xlabel('Lab Measurements', fontsize=12)
    plt.ylabel('Percentage of ESRD Patients (%)', fontsize=12)
    plt.xticks(range(len(results_df)), [f"{row['itemid']}" for _, row in results_df.iterrows()], rotation=45)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    plt.subplot(2, 1, 2)
    x = np.arange(len(results_df))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    bars1 = ax1.bar(x - width/2, results_df['total_measurements'], width, 
                   label='Total Measurements', color='skyblue', alpha=0.7)
    ax1.set_xlabel('Lab Measurements', fontsize=12)
    ax1.set_ylabel('Number of Measurements', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, results_df['percentage_of_esrd_patients'], width,
                   label='% of ESRD Patients', color='lightcoral', alpha=0.7)
    ax2.set_ylabel('Percentage of ESRD Patients (%)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{row['itemid']}\n{row['lab_name'][:30]}..." if len(row['lab_name']) > 30 
                        else f"{row['itemid']}\n{row['lab_name']}" for _, row in results_df.iterrows()], 
                       rotation=45, ha='right')
    
    plt.title('Top 10 Lab Measurements for ESRD Patients: Volume vs Coverage', fontsize=14, fontweight='bold')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/esrd_top_lab_measurements.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_detailed_report(results_df, patients_df, esrd_lab_events):
    
    report = []
    report.append("=" * 80)
    report.append("ESRD PATIENTS - TOP 10 MOST COMMON LAB MEASUREMENTS ANALYSIS")
    report.append("=" * 80)
    report.append("")
    
    report.append(f"Dataset Summary:")
    report.append(f"  - Total ESRD patients: {len(patients_df):,}")
    report.append(f"  - Total lab events for ESRD patients: {len(esrd_lab_events):,}")
    report.append(f"  - Unique lab measurements: {esrd_lab_events['itemid'].nunique():,}")
    report.append(f"  - Average lab events per ESRD patient: {len(esrd_lab_events) / len(patients_df):.1f}")
    report.append("")
    
    report.append("Top 10 Most Common Lab Measurements:")
    report.append("-" * 80)
    
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        report.append(f"{i}. {row['lab_name']} (ItemID: {row['itemid']})")
        report.append(f"   Total measurements: {row['total_measurements']:,} ({row['percentage_of_lab_events']:.2f}% of all lab events)")
        report.append(f"   ESRD patients with this test: {row['patients_with_test']:,} ({row['percentage_of_esrd_patients']:.1f}% of ESRD patients)")
        report.append("")
        
    return "\n".join(report)

def main():
    try:
        patients_df, _, d_labitems_df, labevents_df = load_data()
        
        print("\nAnalyzing top 10 lab measurements...")
        results_df, esrd_lab_events = analyze_top_lab_measurements(patients_df, labevents_df, d_labitems_df)
        
        print("\nTop 10 Most Common Lab Measurements for ESRD Patients:")
        print("=" * 100)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        print(results_df.to_string(index=False))
        
        output_dir = generate_data_path_latest_rep
        print(f"\nCreating visualizations... (saving to {output_dir})")
        create_visualizations(results_df, output_dir)
        
        report = generate_detailed_report(results_df, patients_df, esrd_lab_events)
        print("\n" + report)
        
        report_path = f"{generate_data_path_latest_rep}/esrd_lab_analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nDetailed report saved to: {report_path}")
        
        csv_path = f"{generate_data_path_latest_rep}/esrd_top_lab_measurements.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
