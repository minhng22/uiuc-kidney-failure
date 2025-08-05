#!/usr/bin/env python3
"""
Script to parse model performance from eval_all_rep1.log to eval_all_rep5.log
and calculate mean ± standard deviation for each model.
"""

import re
import os
import numpy as np
from collections import defaultdict
from datetime import datetime

def parse_log_file(log_path):
    """Parse a single log file and extract performance metrics."""
    results = {}
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract repetition number from filename
    rep_match = re.search(r'rep(\d+)', log_path)
    if not rep_match:
        return results
    
    rep_num = rep_match.group(1)
    
    # Define the models and their patterns
    model_patterns = {
        # Cox models - different variants
        'cox_ti': {
            'section': r'Running non-time-variant Cox model evaluation\.\.\.(.*?)(?=Running time-variant Cox|Running heterogeneous Cox|Running egfr raw Cox|✓ cox completed)',
            'c_index': r'Concordance Index Test:\s*([\d.]+)',
            'brier': r'Integrated Brier Score Test:\s*([\d.]+)',
            'auc': r'Mean time-dependent AUC:\s*([\d.]+)'
        },
        'cox_tv': {
            'section': r'Running time-variant Cox model evaluation.*?\.\.\.(.*?)(?=Running heterogeneous Cox|Running egfr raw Cox|✓ cox completed)',
            'c_index': r'Concordance Index Test:\s*([\d.]+)',
            'brier': r'Integrated Brier Score Test:\s*([\d.]+)',
            'auc': r'Mean time-dependent AUC:\s*([\d.]+)'
        },
        'cox_hg': {
            'section': r'Running heterogeneous Cox model evaluation.*?\.\.\.(.*?)(?=Running egfr raw Cox|✓ cox completed)',
            'c_index': r'Concordance Index Test:\s*([\d.]+)',
            'brier': r'Integrated Brier Score Test:\s*([\d.]+)',
            'auc': r'Mean time-dependent AUC:\s*([\d.]+)'
        },
        'cox_egfr': {
            'section': r'Running egfr raw Cox model evaluation.*?\.\.\.(.*?)(?=✓ cox completed)',
            'c_index': r'Concordance Index Test:\s*([\d.]+)',
            'brier': r'Integrated Brier Score Test:\s*([\d.]+)',
            'auc': r'Mean time-dependent AUC:\s*([\d.]+)'
        },
        # DeepSurv
        'deepsurv': {
            'section': r'==================== Running deepsurv.*?====================.*?(.*?)(?=✓ deepsurv completed|✗ deepsurv failed|===================)',
            'c_index': r'C-Index on Test Data:\s*([\d.]+)',
            'brier': r'Integrated Brier Score Test:\s*([\d.]+)',
            'auc': r'Mean AUC:\s*([\d.]+)'
        },
        # GBSA
        'gbsa': {
            'section': r'==================== Running gbsa.*?====================.*?(.*?)(?=✓ gbsa completed|✗ gbsa failed|===================)',
            'c_index': r'Concordance Index Test:\s*([\d.]+)',
            'brier': r'Integrated Brier Score Test:\s*([\d.]+)',
            'auc': r'Mean AUC:\s*([\d.]+)'
        },
        # Hazard Transformer
        'hazard_transformer': {
            'section': r'==================== Running hazard_transformer.*?====================.*?(.*?)(?=✓ hazard_transformer completed|✗ hazard_transformer failed|===================)',
            'c_index': r'C-index:\s*([\d.]+)',
            'brier': r'Integrated Brier Score Test:\s*([\d.]+)',
            'auc': r'Mean time-dependent AUC:\s*([\d.]+)'
        },
        # RNN Survival
        'rnnsurv': {
            'section': r'==================== Running rnnsurv.*?====================.*?(.*?)(?=✓ rnnsurv completed|✗ rnnsurv failed|===================)',
            'c_index': r'C-Index on Test Data:\s*([\d.]+)',
            'brier': r'Integrated Brier Score Test:\s*([\d.]+)',
            'auc': r'Mean time-dependent AUC:\s*([\d.]+)'
        },
        # Survival SVM
        'survival_svm': {
            'section': r'==================== Running survival_svm.*?====================.*?(.*?)(?=✓ survival_svm completed|✗ survival_svm failed|===================)',
            'c_index': r'Concordance Index Test:\s*([\d.]+)',
            'brier': r'Integrated Brier Score Test:\s*([\d.]+)',
            'auc': r'Mean time-dependent AUC:\s*([\d.]+)'
        },
        # Weibull
        'weibul': {
            'section': r'==================== Running weibul.*?====================.*?(.*?)(?=✓ weibul completed|✗ weibul failed|===================)',
            'c_index': r'C-index:\s*([\d.]+)',
            'brier': r'Integrated Brier Score Test:\s*([\d.]+)',
            'auc': r'Mean time-dependent AUC:\s*([\d.]+)'
        }
    }
    
    # Extract metrics for each model
    for model_name, patterns in model_patterns.items():
        try:
            # Find the section for this model
            section_match = re.search(patterns['section'], content, re.DOTALL | re.IGNORECASE)
            if not section_match:
                continue
            
            section_text = section_match.group(1)
            
            # Extract metrics from the section
            metrics = {}
            
            # C-index
            c_index_match = re.search(patterns['c_index'], section_text)
            if c_index_match:
                metrics['c_index'] = float(c_index_match.group(1))
            
            # Brier Score
            brier_match = re.search(patterns['brier'], section_text)
            if brier_match:
                metrics['brier'] = float(brier_match.group(1))
            
            # AUC
            auc_match = re.search(patterns['auc'], section_text)
            if auc_match:
                metrics['auc'] = float(auc_match.group(1))
            
            if metrics:  # Only add if we found at least one metric
                results[model_name] = metrics
                
        except Exception as e:
            print(f"Warning: Error parsing {model_name} in {log_path}: {e}")
            continue
    
    return results

def calculate_statistics(all_results):
    """Calculate mean and standard deviation for each model and metric."""
    stats = {}
    
    # Get all unique models across all repetitions
    all_models = set()
    for rep_results in all_results.values():
        all_models.update(rep_results.keys())
    
    for model in all_models:
        stats[model] = {}
        
        # Get all metric types for this model
        all_metrics = set()
        for rep_results in all_results.values():
            if model in rep_results:
                all_metrics.update(rep_results[model].keys())
        
        for metric in all_metrics:
            values = []
            for rep_results in all_results.values():
                if model in rep_results and metric in rep_results[model]:
                    values.append(rep_results[model][metric])
            
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                stats[model][metric] = {
                    'mean': mean_val,
                    'std': std_val,
                    'values': values,
                    'n': len(values)
                }
    
    return stats

def main():
    script_dir = "/home/minhn2/uiuc-kidney-failure/pkgs/scripts"
    
    # Parse all log files
    all_results = {}
    for rep in range(1, 6):
        log_path = os.path.join(script_dir, f"eval_all_rep{rep}.log")
        if os.path.exists(log_path):
            print(f"Parsing {log_path}...")
            results = parse_log_file(log_path)
            all_results[f"rep{rep}"] = results
            print(f"  Found {len(results)} models")
        else:
            print(f"Warning: {log_path} not found")
    
    if not all_results:
        print("No log files found!")
        return
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats = calculate_statistics(all_results)
    
    # Generate output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/home/minhn2/uiuc-kidney-failure/model_performance_summary_{timestamp}.log"
    
    with open(output_file, 'w') as f:
        # Sort models for consistent output
        sorted_models = sorted(stats.keys())
        
        # Only output the summary table
        f.write("SUMMARY TABLE\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Model':<20} {'C-Index':<15} {'Brier Score':<15} {'AUC':<15}\n")
        f.write("-" * 80 + "\n")
        
        for model in sorted_models:
            metrics = stats[model]
            
            c_index_str = ""
            if 'c_index' in metrics:
                m, s = metrics['c_index']['mean'], metrics['c_index']['std']
                c_index_str = f"{m:.3f}±{s:.3f}"
            
            brier_str = ""
            if 'brier' in metrics:
                m, s = metrics['brier']['mean'], metrics['brier']['std']
                brier_str = f"{m:.3f}±{s:.3f}"
            
            auc_str = ""
            if 'auc' in metrics:
                m, s = metrics['auc']['mean'], metrics['auc']['std']
                auc_str = f"{m:.3f}±{s:.3f}"
            
            f.write(f"{model:<20} {c_index_str:<15} {brier_str:<15} {auc_str:<15}\n")
    
    print(f"\nResults saved to: {output_file}")
    print("\nBrief summary:")
    for model in sorted(stats.keys()):
        metrics = stats[model]
        if 'c_index' in metrics:
            m, s = metrics['c_index']['mean'], metrics['c_index']['std']
            print(f"  {model}: C-Index = {m:.3f} ± {s:.3f}")

if __name__ == "__main__":
    main()
