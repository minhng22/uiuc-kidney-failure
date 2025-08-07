#!/usr/bin/env python3
"""
Test script to validate SurvTimeSurvival and DeepONet implementations
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pkgs.data.types import ExperimentScenario

def test_survtimesurvival():
    """Test SurvTimeSurvival model implementation"""
    print("Testing SurvTimeSurvival model...")
    
    try:
        from pkgs.models.survtimesurvival import SurvTimeSurvival
        
        # Create a model with the new SurvTRACE-based parameters
        model = SurvTimeSurvival(
            input_dim=3,
            num_categorical_feature=0,
            num_numerical_feature=3,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            hidden_dropout_prob=0.1,
            num_durations=5,
            max_position_embeddings=10
        )
        
        # Test forward pass - numerical features only
        batch_size, seq_len, input_dim = 4, 10, 3
        input_nums = torch.randn(batch_size, seq_len, input_dim)
        seq_lengths = torch.tensor([10, 8, 6, 9])
        
        # Forward pass returns hazard logits
        hazard_logits = model(input_nums=input_nums, seq_lengths=seq_lengths)
        
        print(f"  Input shape: {input_nums.shape}")
        print(f"  Hazard logits shape: {hazard_logits.shape}")
        print(f"  Hazard logits range: [{hazard_logits.min():.3f}, {hazard_logits.max():.3f}]")
        
        # Test hazard prediction
        hazard = model.predict_hazard(input_nums=input_nums, seq_lengths=seq_lengths)
        print(f"  Hazard rates shape: {hazard.shape}")
        print(f"  Hazard rates range: [{hazard.min():.3f}, {hazard.max():.3f}]")
        
        # Test survival prediction
        survival = model.predict_survival(input_nums=input_nums, seq_lengths=seq_lengths)
        print(f"  Survival probs shape: {survival.shape}")
        print(f"  Survival probs range: [{survival.min():.3f}, {survival.max():.3f}]")
        
        # Test risk score prediction
        risk_scores = model.predict_risk_score(input_nums=input_nums, seq_lengths=seq_lengths)
        print(f"  Risk scores shape: {risk_scores.shape}")
        print(f"  Risk scores range: [{risk_scores.min():.3f}, {risk_scores.max():.3f}]")
        
        # Test survival prediction at specific time points
        time_points = torch.tensor([0.1, 0.3, 0.6, 1.0])  # Normalized time points
        surv_at_times = model.predict_survival_at_times(input_nums=input_nums, time_points=time_points, seq_lengths=seq_lengths)
        print(f"  Survival at time points shape: {surv_at_times.shape}")
        
        print("  ✓ SurvTimeSurvival model test passed!")
        
    except Exception as e:
        print(f"  ✗ SurvTimeSurvival model test failed: {e}")
        return False
    
    return True


def test_deeponet():
    """Test DeepONet model implementation"""
    print("Testing DeepONet model...")
    
    try:
        from pkgs.models.deeponet import DeepONet
        
        # Create a simple model
        model = DeepONet(
            input_dim=3,
            branch_hidden_dims=[64, 128],
            trunk_hidden_dims=[64, 128],
            query_dim=1,
            dropout=0.1,
            operator_dim=64
        )
        
        # Test forward pass
        batch_size, seq_len, input_dim = 4, 10, 3
        u = torch.randn(batch_size, seq_len, input_dim)  # Covariate histories
        
        # Test with single query time per sample
        y_single = torch.randn(batch_size, 1)  # Query times
        output_single = model(u, y_single)
        
        print(f"  Input u shape: {u.shape}")
        print(f"  Query y shape: {y_single.shape}")
        print(f"  Output shape (single query): {output_single.shape}")
        
        # Test with multiple query times
        num_queries = 5
        y_multi = torch.randn(num_queries, 1)  # Query times
        output_multi = model(u, y_multi)
        
        print(f"  Multiple queries y shape: {y_multi.shape}")
        print(f"  Output shape (multi query): {output_multi.shape}")
        
        # Test hazard prediction
        hazard_rates = model.predict_hazard(u, y_multi)
        print(f"  Hazard rates shape: {hazard_rates.shape}")
        print(f"  Hazard rates range: [{hazard_rates.min():.3f}, {hazard_rates.max():.3f}]")
        
        # Test survival prediction
        survival_probs = model.predict_survival(u, y_multi)
        print(f"  Survival probs shape: {survival_probs.shape}")
        print(f"  Survival probs range: [{survival_probs.min():.3f}, {survival_probs.max():.3f}]")
        
        # Test risk score prediction
        risk_scores = model.predict_risk_score(u)
        print(f"  Risk scores shape: {risk_scores.shape}")
        print(f"  Risk scores range: [{risk_scores.min():.3f}, {risk_scores.max():.3f}]")
        
        print("  ✓ DeepONet model test passed!")
        
    except Exception as e:
        print(f"  ✗ DeepONet model test failed: {e}")
        return False
    
    return True


def test_datasets():
    """Test dataset implementations"""
    print("Testing dataset implementations...")
    
    try:
        # Create some dummy data
        np.random.seed(42)
        subjects = np.arange(1, 11)  # 10 subjects
        
        data = []
        for subject_id in subjects:
            n_obs = np.random.randint(1, 6)  # 1-5 observations per subject
            for i in range(n_obs):
                data.append({
                    'subject_id': subject_id,
                    'duration_in_days': 30 + i * 30 + np.random.randint(0, 30),
                    'has_esrd': np.random.choice([0, 1], p=[0.8, 0.2]),
                    'egfr': np.random.normal(60, 20),
                    'age': np.random.randint(30, 80),
                    'gender': np.random.choice([0, 1]),
                    'serum_creatinine': np.random.normal(1.2, 0.5),
                    'protein': np.random.normal(0.15, 0.1),
                    'albumin': np.random.normal(3.5, 0.8),
                    'egfr_missing': 0,
                    'protein_missing': 0,
                    'albumin_missing': 0,
                })
        
        import pandas as pd
        df = pd.DataFrame(data)
        
        # Test SurvTimeSurvival dataset
        from pkgs.experiments.survtimesurvival import SurvTimeSurvivalDataset
        
        for scenario in [ExperimentScenario.TIME_VARIANT, ExperimentScenario.NON_TIME_VARIANT]:
            dataset = SurvTimeSurvivalDataset(df, scenario)
            sample = dataset[0]
            print(f"  SurvTimeSurvival Dataset ({scenario.value}):")
            print(f"    Input nums shape: {sample['input_nums'].shape}")
            print(f"    Duration: {sample['duration'].item():.1f}")
            print(f"    Event: {sample['event'].item()}")
            print(f"    Seq length: {sample['seq_length'].item()}")
        
        # Test DeepONet dataset
        from pkgs.experiments.deeponet import DeepONetDataset
        
        for scenario in [ExperimentScenario.TIME_VARIANT, ExperimentScenario.NON_TIME_VARIANT]:
            dataset = DeepONetDataset(df, scenario)
            sample = dataset[0]
            print(f"  DeepONet Dataset ({scenario.value}):")
            print(f"    Features shape: {sample['features'].shape}")
            print(f"    Query time shape: {sample['query_time'].shape}")
            print(f"    Duration: {sample['duration'].item():.1f}")
            print(f"    Event: {sample['event'].item()}")
        
        print("  ✓ Dataset tests passed!")
        
    except Exception as e:
        print(f"  ✗ Dataset test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests"""
    print("="*50)
    print("Testing SurvTimeSurvival and DeepONet Implementations")
    print("="*50)
    
    tests = [
        test_survtimesurvival,
        test_deeponet,
        test_datasets
    ]
    
    results = []
    for test in tests:
        print()
        result = test()
        results.append(result)
    
    print()
    print("="*50)
    print("Test Summary:")
    print("="*50)
    
    test_names = [
        "SurvTimeSurvival Model",
        "DeepONet Model", 
        "Dataset Implementations"
    ]
    
    all_passed = True
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "PASSED" if result else "FAILED"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    print()
    if all_passed:
        print("✓ All tests passed! Ready to run experiments.")
    else:
        print("✗ Some tests failed. Please check the implementations.")
    
    return all_passed


if __name__ == '__main__':
    main()
