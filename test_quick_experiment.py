#!/usr/bin/env python3
"""
Quick test to verify experiments run without major errors
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from pkgs.data.types import ExperimentScenario

def create_test_data():
    """Create minimal test dataset"""
    data = []
    for subject_id in range(10):  # Small dataset
        for i in range(3):  # Few observations per subject
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
    
    return pd.DataFrame(data)

def test_survtimesurvival_experiment():
    """Test SurvTimeSurvival experiment with minimal setup"""
    print("Testing SurvTimeSurvival experiment...")
    
    try:
        from pkgs.experiments.survtimesurvival import SurvTimeSurvivalDataset, survtime_loss_function
        import torch
        from torch.utils.data import DataLoader
        from pkgs.models.survtimesurvival import SurvTimeSurvival
        
        # Create test data
        df = create_test_data()
        
        # Test dataset
        dataset = SurvTimeSurvivalDataset(df, ExperimentScenario.TIME_VARIANT)
        print(f"  Dataset length: {len(dataset)}")
        
        # Check first sample
        sample = dataset[0]
        print(f"  Sample keys: {sample.keys()}")
        print(f"  Input nums shape: {sample['input_nums'].shape if sample['input_nums'] is not None else 'None'}")
        print(f"  Duration: {sample['duration']}")
        print(f"  Event: {sample['event']}")
        
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Test model
        sample = dataset[0]
        input_dim = sample['input_nums'].shape[-1]
        
        model = SurvTimeSurvival(
            input_dim=input_dim,
            num_numerical_feature=input_dim,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            num_durations=5
        )
        
        # Test forward pass and loss
        batch = next(iter(loader))
        input_nums = batch['input_nums']
        durations = batch['duration']
        events = batch['event']
        seq_lengths = batch['seq_length']
        
        hazard_logits = model(input_nums=input_nums, seq_lengths=seq_lengths)
        loss = survtime_loss_function(hazard_logits, durations, events)
        
        print(f"  Batch size: {input_nums.size(0)}")
        print(f"  Input shape: {input_nums.shape}")
        print(f"  Hazard logits shape: {hazard_logits.shape}")
        print(f"  Loss: {loss.item():.4f}")
        print("  ✓ SurvTimeSurvival experiment test passed!")
        
    except Exception as e:
        print(f"  ✗ SurvTimeSurvival experiment test failed: {e}")
        return False
    
    return True

def test_deeponet_experiment():
    """Test DeepONet experiment with minimal setup"""
    print("\nTesting DeepONet experiment...")
    
    try:
        from pkgs.experiments.deeponet import DeepONetDataset, deeponet_survival_loss
        import torch
        from torch.utils.data import DataLoader
        from pkgs.models.deeponet import DeepONet
        
        # Create test data
        df = create_test_data()
        
        # Test dataset
        dataset = DeepONetDataset(df, ExperimentScenario.TIME_VARIANT)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Test model
        sample = dataset[0]
        input_dim = sample['features'].shape[-1]
        
        model = DeepONet(
            input_dim=input_dim,
            branch_hidden_dims=[32, 32],
            trunk_hidden_dims=[32, 32],
            operator_dim=32
        )
        
        # Test forward pass and loss
        batch = next(iter(loader))
        features = batch['features']
        query_time = batch['query_time']
        durations = batch['duration']
        events = batch['event']
        
        output = model(features, query_time)
        survival_probs = model.predict_survival(features, query_time)
        loss = deeponet_survival_loss(model, features, query_time, durations, events)
        
        print(f"  Batch size: {features.size(0)}")
        print(f"  Features shape: {features.shape}")
        print(f"  Query time shape: {query_time.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Loss: {loss.item():.4f}")
        print("  ✓ DeepONet experiment test passed!")
        
    except Exception as e:
        print(f"  ✗ DeepONet experiment test failed: {e}")
        return False
    
    return True

if __name__ == '__main__':
    print("=" * 50)
    print("Quick Experiment Test")
    print("=" * 50)
    
    results = []
    results.append(test_survtimesurvival_experiment())
    results.append(test_deeponet_experiment())
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    print(f"SurvTimeSurvival Experiment: {'PASSED' if results[0] else 'FAILED'}")
    print(f"DeepONet Experiment: {'PASSED' if results[1] else 'FAILED'}")
    
    if all(results):
        print("\n✓ All experiment tests passed! Ready for full training.")
    else:
        print("\n✗ Some experiment tests failed.")
