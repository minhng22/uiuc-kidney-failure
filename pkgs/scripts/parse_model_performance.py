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
        # Survival SVM
        'survival_svm': {
            'section': r'==================== Running survival_svm.*?====================.*?(.*?)(?=✓ survival_svm completed|✗ survival_svm failed|===================)',
            'c_index': r'Concordance Index Test:\s*([\d.]+)',
            'brier': r'Integrated Brier Score Test:\s*([\d.]+)',
            'auc': r'Mean AUC:\s*([\d.]+)'
        },
        # Weibull
        'weibul': {
            'section': r'==================== Running weibul.*?====================.*?(.*?)(?=✓ weibul completed|✗ weibul failed|===================)',
            'c_index': r'C-index:\s*([\d.]+)',
            'brier': r'Integrated Brier Score Test:\s*([\d.]+)',
            'auc': r'Mean time-dependent AUC:\s*([\d.]+)'
        },
        # LogisticHazard
        'logistic_hazard': {
            'section': r'==================== Running logistic_hazard.*?====================.*?(.*?)(?=✓ logistic_hazard completed|✗ logistic_hazard failed|===================)',
            'c_index': r'Global test C-index:\s*([\d.]+)',
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
    
    # Special handling for dynamic_deephit with multiple settings
    ddh_results = parse_dynamic_deephit(content)
    results.update(ddh_results)
    
    # Special handling for hazard_transformer with multiple settings
    ht_results = parse_hazard_transformer(content)
    results.update(ht_results)
    
    # Special handling for rnnsurv with multiple settings
    rnn_results = parse_rnnsurv(content)
    results.update(rnn_results)
    
    # Special handling for logistic_hazard with multiple settings
    lh_results = parse_logistic_hazard(content)
    results.update(lh_results)
    
    return results

def parse_dynamic_deephit(content):
    """Special parser for dynamic_deephit which has multiple settings in one section."""
    ddh_results = {}
    
    # Find the entire dynamic_deephit section
    ddh_section_match = re.search(
        r'==================== Running dynamic_deephit.*?====================.*?(.*?)(?=✓ dynamic_deephit completed|✗ dynamic_deephit failed|===================)',
        content, re.DOTALL | re.IGNORECASE
    )
    
    if not ddh_section_match:
        return ddh_results
    
    ddh_section = ddh_section_match.group(1)
    
    # Define the three settings and their patterns
    settings = {
        'ddh_egfr_tv': {
            'data_path_pattern': r'egfr_tv_train_data\.csv.*?egfr_tv_test_data\.csv',
            'start_marker': r'egfr_tv_train_data\.csv',
            'end_marker': r'(?=Train data path.*?heterogen_train_data\.csv|Train data path.*?egfr_components_train_data\.csv|✓ dynamic_deephit completed|$)'
        },
        'ddh_heterogen': {
            'data_path_pattern': r'heterogen_train_data\.csv.*?heterogen_test_data\.csv',
            'start_marker': r'heterogen_train_data\.csv',
            'end_marker': r'(?=Train data path.*?egfr_components_train_data\.csv|✓ dynamic_deephit completed|$)'
        },
        'ddh_egfr_components': {
            'data_path_pattern': r'egfr_components_train_data\.csv.*?egfr_components_test_data\.csv',
            'start_marker': r'egfr_components_train_data\.csv',
            'end_marker': r'(?=✓ dynamic_deephit completed|$)'
        }
    }
    
    for setting_name, patterns in settings.items():
        try:
            # Find the start position of this setting
            start_match = re.search(patterns['start_marker'], ddh_section)
            if not start_match:
                continue
            
            # Extract the subsection for this setting
            start_pos = start_match.start()
            end_match = re.search(patterns['end_marker'], ddh_section[start_pos:])
            
            if end_match:
                end_pos = start_pos + end_match.start()
                setting_section = ddh_section[start_pos:end_pos]
            else:
                setting_section = ddh_section[start_pos:]
            
            # Extract metrics from this setting section
            metrics = {}
            
            # C-index (look for "Global test C-index")
            c_index_match = re.search(r'Global test C-index.*?:\s*([\d.]+)', setting_section)
            if c_index_match:
                metrics['c_index'] = float(c_index_match.group(1))
            
            # Brier Score
            brier_match = re.search(r'Integrated Brier Score Test:\s*([\d.]+)', setting_section)
            if brier_match:
                metrics['brier'] = float(brier_match.group(1))
            
            # AUC
            auc_match = re.search(r'Mean time-dependent AUC:\s*([\d.]+)', setting_section)
            if auc_match:
                metrics['auc'] = float(auc_match.group(1))
            
            if metrics:  # Only add if we found at least one metric
                ddh_results[setting_name] = metrics
                
        except Exception as e:
            print(f"Warning: Error parsing {setting_name}: {e}")
            continue
    
    return ddh_results

def parse_hazard_transformer(content):
    """Special parser for hazard_transformer which has multiple settings in one section."""
    ht_results = {}
    
    # Find the entire hazard_transformer section
    ht_section_match = re.search(
        r'==================== Running hazard_transformer.*?====================.*?(.*?)(?=✓ hazard_transformer completed|✗ hazard_transformer failed|===================)',
        content, re.DOTALL | re.IGNORECASE
    )
    
    if not ht_section_match:
        return ht_results
    
    ht_section = ht_section_match.group(1)
    
    # Define the three settings and their patterns
    settings = {
        'hazard_transformer_egfr_tv': {
            'data_path_pattern': r'egfr_tv_train_data\.csv.*?egfr_tv_test_data\.csv',
            'start_marker': r'egfr_tv_train_data\.csv',
            'end_marker': r'(?=Train data path.*?heterogen_train_data\.csv|Train data path.*?egfr_components_train_data\.csv|✓ hazard_transformer completed|$)'
        },
        'hazard_transformer_heterogen': {
            'data_path_pattern': r'heterogen_train_data\.csv.*?heterogen_test_data\.csv',
            'start_marker': r'heterogen_train_data\.csv',
            'end_marker': r'(?=Train data path.*?egfr_components_train_data\.csv|✓ hazard_transformer completed|$)'
        },
        'hazard_transformer_egfr_components': {
            'data_path_pattern': r'egfr_components_train_data\.csv.*?egfr_components_test_data\.csv',
            'start_marker': r'egfr_components_train_data\.csv',
            'end_marker': r'(?=✓ hazard_transformer completed|$)'
        }
    }
    
    for setting_name, patterns in settings.items():
        try:
            # Find the start position of this setting
            start_match = re.search(patterns['start_marker'], ht_section)
            if not start_match:
                continue
            
            # Extract the subsection for this setting
            start_pos = start_match.start()
            end_match = re.search(patterns['end_marker'], ht_section[start_pos:])
            
            if end_match:
                end_pos = start_pos + end_match.start()
                setting_section = ht_section[start_pos:end_pos]
            else:
                setting_section = ht_section[start_pos:]
            
            # Extract metrics from this setting section
            metrics = {}
            
            # C-index (look for "C-index:")
            c_index_match = re.search(r'C-index:\s*([\d.]+)', setting_section)
            if c_index_match:
                metrics['c_index'] = float(c_index_match.group(1))
            
            # Brier Score
            brier_match = re.search(r'Integrated Brier Score Test:\s*([\d.]+)', setting_section)
            if brier_match:
                metrics['brier'] = float(brier_match.group(1))
            
            # AUC
            auc_match = re.search(r'Mean time-dependent AUC:\s*([\d.]+)', setting_section)
            if auc_match:
                metrics['auc'] = float(auc_match.group(1))
            
            if metrics:  # Only add if we found at least one metric
                ht_results[setting_name] = metrics
                
        except Exception as e:
            print(f"Warning: Error parsing {setting_name}: {e}")
            continue
    
    return ht_results

def parse_rnnsurv(content):
    """Special parser for rnnsurv which has multiple settings in one section."""
    rnn_results = {}
    
    # Find the entire rnnsurv section
    rnn_section_match = re.search(
        r'==================== Running rnnsurv.*?====================.*?(.*?)(?=✓ rnnsurv completed|✗ rnnsurv failed|===================)',
        content, re.DOTALL | re.IGNORECASE
    )
    
    if not rnn_section_match:
        return rnn_results
    
    rnn_section = rnn_section_match.group(1)
    
    # Define the three settings and their patterns
    settings = {
        'rnnsurv_egfr_tv': {
            'data_path_pattern': r'egfr_tv_train_data\.csv.*?egfr_tv_test_data\.csv',
            'start_marker': r'egfr_tv_train_data\.csv',
            'end_marker': r'(?=Train data path.*?heterogen_train_data\.csv|Train data path.*?egfr_components_train_data\.csv|✓ rnnsurv completed|$)'
        },
        'rnnsurv_heterogen': {
            'data_path_pattern': r'heterogen_train_data\.csv.*?heterogen_test_data\.csv',
            'start_marker': r'heterogen_train_data\.csv',
            'end_marker': r'(?=Train data path.*?egfr_components_train_data\.csv|✓ rnnsurv completed|$)'
        },
        'rnnsurv_egfr_components': {
            'data_path_pattern': r'egfr_components_train_data\.csv.*?egfr_components_test_data\.csv',
            'start_marker': r'egfr_components_train_data\.csv',
            'end_marker': r'(?=✓ rnnsurv completed|$)'
        }
    }
    
    for setting_name, patterns in settings.items():
        try:
            # Find the start position of this setting
            start_match = re.search(patterns['start_marker'], rnn_section)
            if not start_match:
                continue
            
            # Extract the subsection for this setting
            start_pos = start_match.start()
            end_match = re.search(patterns['end_marker'], rnn_section[start_pos:])
            
            if end_match:
                end_pos = start_pos + end_match.start()
                setting_section = rnn_section[start_pos:end_pos]
            else:
                setting_section = rnn_section[start_pos:]
            
            # Extract metrics from this setting section
            metrics = {}
            
            # C-index (look for "C-Index on Test Data:")
            c_index_match = re.search(r'C-Index on Test Data:\s*([\d.]+)', setting_section)
            if c_index_match:
                metrics['c_index'] = float(c_index_match.group(1))
            
            # Brier Score
            brier_match = re.search(r'Integrated Brier Score Test:\s*([\d.]+)', setting_section)
            if brier_match:
                metrics['brier'] = float(brier_match.group(1))
            
            # AUC
            auc_match = re.search(r'Mean time-dependent AUC:\s*([\d.]+)', setting_section)
            if auc_match:
                metrics['auc'] = float(auc_match.group(1))
            
            if metrics:  # Only add if we found at least one metric
                rnn_results[setting_name] = metrics
                
        except Exception as e:
            print(f"Warning: Error parsing {setting_name}: {e}")
            continue
    
    return rnn_results

def parse_logistic_hazard(content):
    """Special parser for logistic_hazard which has multiple settings in one section."""
    lh_results = {}
    
    # Find the entire logistic_hazard section
    lh_section_match = re.search(
        r'==================== Running logistic_hazard.*?====================.*?(.*?)(?=✓ logistic_hazard completed|✗ logistic_hazard failed|===================)',
        content, re.DOTALL | re.IGNORECASE
    )
    
    if not lh_section_match:
        return lh_results
    
    lh_section = lh_section_match.group(1)
    
    # Define the three settings and their patterns
    settings = {
        'logistic_hazard_egfr_tv': {
            'data_path_pattern': r'TIME_VARIANT',
            'start_marker': r'=== Evaluation Results for TIME_VARIANT ===',
            'end_marker': r'(?==== Evaluation Results for HETEROGENEOUS ===|=== Evaluation Results for EGFR_COMPONENTS ===|✓ logistic_hazard completed|$)'
        },
        'logistic_hazard_heterogen': {
            'data_path_pattern': r'HETEROGENEOUS',
            'start_marker': r'=== Evaluation Results for HETEROGENEOUS ===',
            'end_marker': r'(?==== Evaluation Results for EGFR_COMPONENTS ===|✓ logistic_hazard completed|$)'
        },
        'logistic_hazard_egfr_components': {
            'data_path_pattern': r'EGFR_COMPONENTS',
            'start_marker': r'=== Evaluation Results for EGFR_COMPONENTS ===',
            'end_marker': r'(?=✓ logistic_hazard completed|$)'
        }
    }
    
    for setting_name, patterns in settings.items():
        try:
            # Find the start position of this setting
            start_match = re.search(patterns['start_marker'], lh_section)
            if not start_match:
                continue
            
            # Extract the subsection for this setting
            start_pos = start_match.start()
            end_match = re.search(patterns['end_marker'], lh_section[start_pos:])
            
            if end_match:
                end_pos = start_pos + end_match.start()
                setting_section = lh_section[start_pos:end_pos]
            else:
                setting_section = lh_section[start_pos:]
            
            # Extract metrics from this setting section
            metrics = {}
            
            # C-index (look for "Global test C-index")
            c_index_match = re.search(r'Global test C-index.*?:\s*([\d.]+)', setting_section)
            if c_index_match:
                metrics['c_index'] = float(c_index_match.group(1))
            
            # Brier Score
            brier_match = re.search(r'Integrated Brier Score Test:\s*([\d.]+)', setting_section)
            if brier_match:
                metrics['brier'] = float(brier_match.group(1))
            
            # AUC
            auc_match = re.search(r'Mean time-dependent AUC:\s*([\d.]+)', setting_section)
            if auc_match:
                metrics['auc'] = float(auc_match.group(1))
            
            if metrics:  # Only add if we found at least one metric
                lh_results[setting_name] = metrics
                
        except Exception as e:
            print(f"Warning: Error parsing {setting_name}: {e}")
            continue
    
    return lh_results

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
    output_file = f"/home/minhn2/uiuc-kidney-failure/pkgs/scripts/model_performance_summary.log"
    
    with open(output_file, 'w') as f:
        # Sort models for consistent output
        sorted_models = sorted(stats.keys())
        
        # Only output the summary table
        f.write("SUMMARY TABLE\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'Model':<40} {'C-Index':<15} {'Brier Score':<15} {'AUC':<15}\n")
        f.write("-" * 90 + "\n")
        
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
            
            f.write(f"{model:<40} {c_index_str:<15} {brier_str:<15} {auc_str:<15}\n")
    
    print(f"\nResults saved to: {output_file}")
    print("\nBrief summary:")
    for model in sorted(stats.keys()):
        metrics = stats[model]
        if 'c_index' in metrics:
            m, s = metrics['c_index']['mean'], metrics['c_index']['std']
            print(f"  {model}: C-Index = {m:.3f} ± {s:.3f}")

if __name__ == "__main__":
    main()
