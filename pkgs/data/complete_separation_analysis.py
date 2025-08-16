#!/usr/bin/env python3
"""
Complete Separation Analysis for Lab Measurements
Analyzes variance of lab measurements conditioned on ESRD event occurrence
to detect potential complete separation issues in survival models.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def analyze_complete_separation(train_path, test_path, output_file):
    """
    Analyze complete separation for all lab measurements in train and test datasets.
    
    Args:
        train_path (str): Path to training data CSV
        test_path (str): Path to test data CSV
        output_file (str): Path to output text file for results
    """
    
    print(f"Starting complete separation analysis...")
    print(f"Train data: {train_path}")
    print(f"Test data: {test_path}")
    
    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv(train_path, index_col=0)
    test_df = pd.read_csv(test_path, index_col=0)
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Identify lab measurement columns (excluding missing indicators and metadata)
    lab_measurements = []
    for col in train_df.columns:
        if col not in ['subject_id', 'duration_in_days', 'start', 'stop', 'has_esrd'] and not col.endswith('_missing'):
            lab_measurements.append(col)
    
    print(f"Found {len(lab_measurements)} lab measurements: {lab_measurements}")
    
    # Open output file
    with open(output_file, 'w') as f:
        f.write("Complete Separation Analysis for Lab Measurements\n")
        f.write("=" * 50 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Train data: {train_path}\n")
        f.write(f"Test data: {test_path}\n")
        f.write(f"Train data shape: {train_df.shape}\n")
        f.write(f"Test data shape: {test_df.shape}\n")
        f.write(f"Lab measurements analyzed: {len(lab_measurements)}\n\n")
        
        # Analyze each dataset
        for dataset_name, df in [("TRAINING", train_df), ("TEST", test_df)]:
            f.write(f"\n{dataset_name} DATASET ANALYSIS\n")
            f.write("=" * 30 + "\n")
            
            # Convert has_esrd to boolean
            events = df['has_esrd'].astype(bool)
            
            # Basic statistics
            n_total = len(df)
            n_events = events.sum()
            n_no_events = n_total - n_events
            event_rate = n_events / n_total * 100
            
            f.write(f"Total records: {n_total:,}\n")
            f.write(f"ESRD events: {n_events:,} ({event_rate:.2f}%)\n")
            f.write(f"Non-ESRD: {n_no_events:,} ({100-event_rate:.2f}%)\n\n")
            
            # Analyze each lab measurement
            separation_issues = []
            
            for lab in lab_measurements:
                f.write(f"Lab Measurement: {lab}\n")
                f.write("-" * 20 + "\n")
                
                # Get non-missing values only
                missing_col = f"{lab}_missing"
                if missing_col in df.columns:
                    non_missing_mask = df[missing_col] == 0
                    lab_values = df.loc[non_missing_mask, lab]
                    lab_events = events[non_missing_mask]
                else:
                    # If no missing indicator, use all values that are not NaN
                    non_missing_mask = df[lab].notna()
                    lab_values = df.loc[non_missing_mask, lab]
                    lab_events = events[non_missing_mask]
                
                n_non_missing = len(lab_values)
                n_missing = n_total - n_non_missing
                
                f.write(f"  Non-missing values: {n_non_missing:,} ({n_non_missing/n_total*100:.2f}%)\n")
                f.write(f"  Missing values: {n_missing:,} ({n_missing/n_total*100:.2f}%)\n")
                
                if n_non_missing == 0:
                    f.write(f"  WARNING: No non-missing values for {lab}\n\n")
                    continue
                
                # Separate values by event status
                event_values = lab_values[lab_events]
                no_event_values = lab_values[~lab_events]
                
                f.write(f"  Values with ESRD: {len(event_values):,}\n")
                f.write(f"  Values without ESRD: {len(no_event_values):,}\n")
                
                if len(event_values) == 0 or len(no_event_values) == 0:
                    f.write(f"  CRITICAL: Complete separation detected - one group has no values!\n")
                    separation_issues.append(f"{lab}: Complete separation - one group empty")
                    f.write("\n")
                    continue
                
                # Calculate variances
                var_with_event = event_values.var()
                var_without_event = no_event_values.var()
                
                # Calculate means and other statistics
                mean_with_event = event_values.mean()
                mean_without_event = no_event_values.mean()
                
                f.write(f"  Mean with ESRD: {mean_with_event:.6f}\n")
                f.write(f"  Mean without ESRD: {mean_without_event:.6f}\n")
                f.write(f"  Variance with ESRD: {var_with_event:.6f}\n")
                f.write(f"  Variance without ESRD: {var_without_event:.6f}\n")
                
                # Check for low variance (potential separation issues)
                low_var_threshold = 1e-6
                variance_ratio = min(var_with_event, var_without_event) / max(var_with_event, var_without_event) if max(var_with_event, var_without_event) > 0 else 0
                
                f.write(f"  Variance ratio (min/max): {variance_ratio:.6f}\n")
                
                # Check for potential issues
                if var_with_event < low_var_threshold or var_without_event < low_var_threshold:
                    issue_msg = f"Very low variance detected (threshold: {low_var_threshold})"
                    f.write(f"  WARNING: {issue_msg}\n")
                    separation_issues.append(f"{lab}: {issue_msg}")
                
                if variance_ratio < 0.001:  # One variance is 1000x smaller than the other
                    issue_msg = f"Extreme variance difference (ratio: {variance_ratio:.8f})"
                    f.write(f"  WARNING: {issue_msg}\n")
                    separation_issues.append(f"{lab}: {issue_msg}")
                
                # Check for identical values within groups
                unique_event_values = len(event_values.unique())
                unique_no_event_values = len(no_event_values.unique())
                
                f.write(f"  Unique values with ESRD: {unique_event_values}\n")
                f.write(f"  Unique values without ESRD: {unique_no_event_values}\n")
                
                if unique_event_values == 1 or unique_no_event_values == 1:
                    issue_msg = "Constant values within group"
                    f.write(f"  WARNING: {issue_msg}\n")
                    separation_issues.append(f"{lab}: {issue_msg}")
                
                # Check ranges
                min_with = event_values.min()
                max_with = event_values.max()
                min_without = no_event_values.min()
                max_without = no_event_values.max()
                
                f.write(f"  Range with ESRD: [{min_with:.6f}, {max_with:.6f}]\n")
                f.write(f"  Range without ESRD: [{min_without:.6f}, {max_without:.6f}]\n")
                
                # Check for non-overlapping ranges
                if max_with < min_without or max_without < min_with:
                    issue_msg = "Non-overlapping ranges (complete separation)"
                    f.write(f"  CRITICAL: {issue_msg}\n")
                    separation_issues.append(f"{lab}: {issue_msg}")
                
                f.write("\n")
            
            # Summary of issues for this dataset
            f.write(f"SUMMARY - {dataset_name} DATASET ISSUES\n")
            f.write("-" * 30 + "\n")
            if separation_issues:
                f.write(f"Found {len(separation_issues)} potential separation issues:\n")
                for i, issue in enumerate(separation_issues, 1):
                    f.write(f"{i}. {issue}\n")
            else:
                f.write("No significant separation issues detected.\n")
            f.write("\n")
        
        f.write("\nANALYSIS COMPLETE\n")
        f.write("=" * 20 + "\n")
        f.write(f"Results saved to: {output_file}\n")
    
    print(f"Analysis complete! Results saved to: {output_file}")

if __name__ == "__main__":
    # Set paths
    train_path = "/home/minhn2/uiuc-kidney-failure/generated_data/rep1/fivelabms_train_data.csv"
    test_path = "/home/minhn2/uiuc-kidney-failure/generated_data/rep1/fivelabms_test_data.csv"
    output_file = "/home/minhn2/uiuc-kidney-failure/complete_separation_analysis_results.txt"
    
    # Run analysis
    try:
        analyze_complete_separation(train_path, test_path, output_file)
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
