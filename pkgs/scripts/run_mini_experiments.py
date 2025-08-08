#!/usr/bin/env python3
"""
Mini experiments for SurvTimeSurvival and DeepONet models
Uses sample datasets and reduced training for quick validation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

from pkgs.experiments.survtimesurvival import SurvTimeSurvivalDataset, survtime_loss_function, get_features_for_scenario
from pkgs.experiments.deeponet import DeepONetDataset, deeponet_survival_loss
from pkgs.models.survtimesurvival import SurvTimeSurvival  
from pkgs.models.deeponet import DeepONet
from pkgs.data.model_data_store import get_train_test_data, sample
from pkgs.data.types import ExperimentScenario
from pkgs.experiments.utils import compute_brier_score_from_risk_scores, round_metric

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mini_survtimesurvival_experiment(scenario: ExperimentScenario):
    """Mini experiment for SurvTimeSurvival model"""
    print(f"\n{'='*50}")
    print(f"Mini SurvTimeSurvival Experiment: {scenario.value}")
    print(f"{'='*50}")
    
    device = get_device()
    
    # Load and sample data
    df, df_test = get_train_test_data(scenario)
    df_sample = sample(df)
    df_test_sample = sample(df_test)
    
    print(f"Sampled data: Train {len(df_sample)}, Test {len(df_test_sample)}")
    
    # Create datasets
    train_dataset = SurvTimeSurvivalDataset(df_sample, scenario, max_seq_length=20)  # Shorter sequences
    test_dataset = SurvTimeSurvivalDataset(df_test_sample, scenario, max_seq_length=20)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model parameters
    features = get_features_for_scenario(scenario)
    input_dim = len(features)
    print(f"Features: {features}, Input dim: {input_dim}")
    
    # Initialize model with smaller architecture
    model = SurvTimeSurvival(
        input_dim=input_dim,
        num_numerical_feature=input_dim,
        hidden_size=32,        # Smaller
        num_hidden_layers=2,   # Fewer layers
        num_attention_heads=4,
        intermediate_size=64,  # Smaller
        hidden_dropout_prob=0.1,
        num_durations=5
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training - few epochs
    print("Training...")
    model.train()
    for epoch in range(3):  # Only 3 epochs
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            input_nums = batch['input_nums'].to(device)
            durations = batch['duration'].to(device)
            events = batch['event'].to(device)
            seq_lengths = batch['seq_length'].to(device)
            
            optimizer.zero_grad()
            hazard_logits = model(input_nums=input_nums, seq_lengths=seq_lengths)
            loss = survtime_loss_function(hazard_logits, durations, events)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0:
            print(f"Epoch {epoch+1}, Loss: {epoch_loss/num_batches:.4f}")
    
    # Evaluation
    print("Evaluating...")
    model.eval()
    test_risk_scores = []
    test_durations = []
    test_events = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_nums = batch['input_nums'].to(device)
            durations = batch['duration']
            events = batch['event']
            seq_lengths = batch['seq_length'].to(device)
            
            risk_scores = model.predict_risk_score(input_nums=input_nums, seq_lengths=seq_lengths)
            
            test_risk_scores.extend(risk_scores.cpu().numpy())
            test_durations.extend(durations.numpy())
            test_events.extend(events.numpy())
    
    # Compute metrics
    test_risk_scores = np.array(test_risk_scores)
    test_durations = np.array(test_durations)
    test_events = np.array(test_events)
    
    # C-Index
    c_index = concordance_index(test_durations, test_risk_scores, test_events)
    print(f"C-Index: {round_metric(c_index)}")
    
    # Brier Score
    try:
        brier_score = compute_brier_score_from_risk_scores(df_sample, df_test_sample, test_risk_scores)
        print(f"Integrated Brier Score: {round_metric(brier_score) if brier_score else 'N/A'}")
    except Exception as e:
        print(f"Brier Score: Error - {e}")
    
    # Time-dependent AUC
    try:
        times = np.linspace(1, 365, 10)  # Fewer time points
        y_train = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df_sample)
        y_test = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df_test_sample)
        
        _, mean_auc = cumulative_dynamic_auc(y_train, y_test, test_risk_scores, times)
        print(f"Mean time-dependent AUC: {round_metric(mean_auc)}")
    except Exception as e:
        print(f"Time-dependent AUC: Error - {e}")
    
    return {
        'scenario': scenario.value,
        'model': 'SurvTimeSurvival',
        'c_index': round_metric(c_index),
        'samples': len(df_test_sample)
    }

def mini_deeponet_experiment(scenario: ExperimentScenario):
    """Mini experiment for DeepONet model"""
    print(f"\n{'='*50}")
    print(f"Mini DeepONet Experiment: {scenario.value}")
    print(f"{'='*50}")
    
    device = get_device()
    
    # Load and sample data
    df, df_test = get_train_test_data(scenario)
    df_sample = sample(df)
    df_test_sample = sample(df_test)
    
    print(f"Sampled data: Train {len(df_sample)}, Test {len(df_test_sample)}")
    
    # Create datasets
    train_dataset = DeepONetDataset(df_sample, scenario, max_seq_length=20)  # Shorter sequences
    test_dataset = DeepONetDataset(df_test_sample, scenario, max_seq_length=20)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model parameters
    features = get_features_for_scenario(scenario)
    input_dim = len(features)
    print(f"Features: {features}, Input dim: {input_dim}")
    
    # Initialize model with smaller architecture
    model = DeepONet(
        input_dim=input_dim,
        branch_hidden_dims=[16, 16],  # Smaller
        trunk_hidden_dims=[16, 16],   # Smaller
        query_dim=1,
        dropout=0.1,
        operator_dim=16               # Smaller
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training - few epochs
    print("Training...")
    model.train()
    for epoch in range(3):  # Only 3 epochs
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            query_times = batch['query_time'].to(device)
            durations = batch['duration'].to(device)
            events = batch['event'].to(device)
            
            query_times = query_times.squeeze(1)  # (batch_size, 1)
            
            optimizer.zero_grad()
            loss = deeponet_survival_loss(model, features, query_times, durations, events)
            
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0:
            print(f"Epoch {epoch+1}, Loss: {epoch_loss/num_batches:.4f}")
    
    # Evaluation
    print("Evaluating...")
    model.eval()
    test_risk_scores = []
    test_durations = []
    test_events = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            query_times = batch['query_time'].to(device)
            durations = batch['duration']
            events = batch['event']
            
            query_times = query_times.squeeze(1)
            
            # Get risk scores using forward pass
            risk_scores = model(features, query_times).squeeze(-1)
            
            test_risk_scores.extend(risk_scores.cpu().numpy())
            test_durations.extend(durations.numpy())
            test_events.extend(events.numpy())
    
    # Compute metrics
    test_risk_scores = np.array(test_risk_scores)
    test_durations = np.array(test_durations)
    test_events = np.array(test_events)
    
    # C-Index
    c_index = concordance_index(test_durations, test_risk_scores, test_events)
    print(f"C-Index: {round_metric(c_index)}")
    
    # Brier Score
    try:
        brier_score = compute_brier_score_from_risk_scores(df_sample, df_test_sample, test_risk_scores)
        print(f"Integrated Brier Score: {round_metric(brier_score) if brier_score else 'N/A'}")
    except Exception as e:
        print(f"Brier Score: Error - {e}")
    
    # Time-dependent AUC
    try:
        times = np.linspace(1, 365, 10)  # Fewer time points
        y_train = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df_sample)
        y_test = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df_test_sample)
        
        _, mean_auc = cumulative_dynamic_auc(y_train, y_test, test_risk_scores, times)
        print(f"Mean time-dependent AUC: {round_metric(mean_auc)}")
    except Exception as e:
        print(f"Time-dependent AUC: Error - {e}")
    
    return {
        'scenario': scenario.value,
        'model': 'DeepONet',
        'c_index': round_metric(c_index),
        'samples': len(df_test_sample)
    }

def run_all_mini_experiments():
    """Run mini experiments for both models on all scenarios"""
    print("🚀 Running Mini Experiments for SurvTimeSurvival and DeepONet")
    print("="*80)
    
    scenarios = [
        ExperimentScenario.NON_TIME_VARIANT,
        ExperimentScenario.TIME_VARIANT,
        ExperimentScenario.HETEROGENEOUS,
        ExperimentScenario.EGFR_COMPONENTS
    ]
    
    results = []
    
    for scenario in scenarios:
        try:
            # SurvTimeSurvival experiment
            result1 = mini_survtimesurvival_experiment(scenario)
            results.append(result1)
            
            # DeepONet experiment  
            result2 = mini_deeponet_experiment(scenario)
            results.append(result2)
            
        except Exception as e:
            print(f"❌ Error in {scenario.value}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 MINI EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Scenario':<20} {'Model':<20} {'C-Index':<10} {'Samples':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['scenario']:<20} {result['model']:<20} {result['c_index']:<10} {result['samples']:<10}")
    
    print(f"\n✅ Mini experiments completed! Both models tested on {len(scenarios)} scenarios.")
    return results

if __name__ == '__main__':
    results = run_all_mini_experiments()
