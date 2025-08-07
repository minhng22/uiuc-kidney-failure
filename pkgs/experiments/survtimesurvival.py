import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

from pkgs.models.survtimesurvival import SurvTimeSurvival
from pkgs.data.model_data_store import get_train_test_data
from pkgs.experiments.utils import round_metric, ex_optuna, get_tv_rnn_model_features, compute_brier_score_from_risk_scores
from pkgs.commons import (
    egfr_tv_survtimesurvival_model_path, hg_survtimesurvival_model_path, 
    egfr_components_survtimesurvival_model_path, egfr_ti_survtimesurvival_model_path
)
from pkgs.data.types import ExperimentScenario
import os


def get_features_for_scenario(scenario_name: ExperimentScenario):
    """Get features for a given scenario, handling cases where get_tv_rnn_model_features returns None"""
    features = get_tv_rnn_model_features(scenario_name)
    if features is None:
        # Handle NON_TIME_VARIANT case which isn't covered by get_tv_rnn_model_features
        if scenario_name == ExperimentScenario.NON_TIME_VARIANT:
            features = ['egfr']  # Only egfr for non-time-variant scenario
        else:
            raise ValueError(f"No features defined for scenario: {scenario_name}")
    return features


class SurvTimeSurvivalDataset(Dataset):
    def __init__(self, df, scenario_name: ExperimentScenario, max_seq_length=50):
        self.df = df
        self.scenario_name = scenario_name
        self.max_seq_length = max_seq_length
        
        # Get features based on scenario
        self.features = get_features_for_scenario(scenario_name)
        
        # Determine categorical vs numerical features
        self.categorical_features = []
        self.numerical_features = self.features.copy()
        
        # For simplicity, treat all features as numerical (can be enhanced later)
        self.num_categorical_feature = 0
        self.num_numerical_feature = len(self.features)
        
        # Group by subject_id to create sequences
        self.subjects = df['subject_id'].unique()
        
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        subject_data = self.df[self.df['subject_id'] == subject_id].sort_values('duration_in_days')
        
        # Extract features for this subject
        if self.scenario_name == ExperimentScenario.NON_TIME_VARIANT:
            # For time-invariant, use the last observation but create a sequence of length 1
            features = subject_data[self.features].iloc[-1:].values.astype(np.float32)
            seq_length = 1
            
            # Pad to max_seq_length
            if self.max_seq_length > 1:
                padding = np.zeros((self.max_seq_length - 1, features.shape[1]), dtype=np.float32)
                features = np.vstack([features, padding])
        else:
            # For time-varying scenarios, use all observations
            features = subject_data[self.features].values.astype(np.float32)
            seq_length = min(len(features), self.max_seq_length)
            
            # Pad or truncate to max_seq_length
            if len(features) > self.max_seq_length:
                features = features[-self.max_seq_length:]  # Keep last max_seq_length observations
                seq_length = self.max_seq_length
            elif len(features) < self.max_seq_length:
                # Pad with zeros
                padding = np.zeros((self.max_seq_length - len(features), features.shape[1]), dtype=np.float32)
                features = np.vstack([features, padding])
        
        # Get outcome data (use the last observation)
        duration = subject_data['duration_in_days'].iloc[-1]
        event = subject_data['has_esrd'].iloc[-1]
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),  # (max_seq_length, num_features)
            'duration': torch.tensor(duration, dtype=torch.float32),
            'event': torch.tensor(event, dtype=torch.float32),
            'seq_length': torch.tensor(seq_length, dtype=torch.long),
            'input_nums': torch.tensor(features, dtype=torch.float32),  # For SurvTRACE compatibility
        }


def survtime_loss_function(hazard_logits, durations, events):
    """
    SurvTRACE-style loss function for discrete time survival analysis
    Based on negative log-likelihood with piecewise constant hazard
    """
    batch_size = hazard_logits.size(0)
    num_durations = hazard_logits.size(1)
    
    # Convert durations to discrete time indices
    max_duration = 365 * 2  # Assume 2 years max follow-up
    duration_cuts = torch.linspace(0, max_duration, num_durations + 1)
    
    # Find which interval each duration falls into
    duration_indices = torch.searchsorted(duration_cuts[1:], durations, right=True)
    duration_indices = torch.clamp(duration_indices, 0, num_durations - 1)
    
    # Calculate loss for each sample
    total_loss = 0.0
    
    for i in range(batch_size):
        t_idx = duration_indices[i].long()
        event = events[i]
        
        # Get hazard probabilities using softmax (discrete hazard model)
        hazard_probs = F.softmax(hazard_logits[i], dim=0)
        
        # Survival probability up to time t
        surv_prob = torch.prod(1 - hazard_probs[:t_idx])
        
        if event == 1:  # Event occurred
            # Likelihood: h(t) * S(t-)
            if t_idx < num_durations:
                hazard_at_t = hazard_probs[t_idx]
                likelihood = hazard_at_t * surv_prob
            else:
                likelihood = surv_prob  # If beyond last interval
        else:  # Censored
            # Likelihood: S(t)
            likelihood = surv_prob
        
        # Add negative log-likelihood
        total_loss -= torch.log(likelihood + 1e-8)
    
    return total_loss / batch_size


def objective(trial, scenario_name: ExperimentScenario):
    # Hyperparameter suggestions for SurvTRACE-style model
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_attention_heads = trial.suggest_categorical('num_attention_heads', [4, 8, 16])
    num_hidden_layers = trial.suggest_categorical('num_hidden_layers', [2, 3, 4])
    intermediate_size = trial.suggest_categorical('intermediate_size', [256, 512, 1024])
    hidden_dropout_prob = trial.suggest_float('hidden_dropout_prob', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    num_durations = trial.suggest_categorical('num_durations', [5, 10, 15])
    
    device = get_device()
    
    # Load data
    df, df_test = get_train_test_data(scenario_name)
    
    # Create datasets
    train_dataset = SurvTimeSurvivalDataset(df, scenario_name)
    val_dataset = SurvTimeSurvivalDataset(df_test, scenario_name)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model parameters
    features = get_features_for_scenario(scenario_name)
    input_dim = len(features)
    
    # Initialize model
    model = SurvTimeSurvival(
        input_dim=input_dim,
        num_categorical_feature=0,  # All numerical for now
        num_numerical_feature=input_dim,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=hidden_dropout_prob,
        num_durations=num_durations
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    model.train()
    num_epochs = 50
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            input_nums = batch['input_nums'].to(device)
            durations = batch['duration'].to(device)
            events = batch['event'].to(device)
            seq_lengths = batch['seq_length'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            hazard_logits = model(input_nums=input_nums, seq_lengths=seq_lengths)
            
            # Compute loss
            loss = survtime_loss_function(hazard_logits, durations, events)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0:
            epoch_loss /= num_batches
            
        # Early stopping check could be added here
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
    
    # Validation
    model.eval()
    val_risk_scores = []
    val_durations = []
    val_events = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_nums = batch['input_nums'].to(device)
            durations = batch['duration']
            events = batch['event']
            seq_lengths = batch['seq_length'].to(device)
            
            # Get risk scores using the model's prediction method
            risk_scores = model.predict_risk_score(input_nums=input_nums, seq_lengths=seq_lengths)
            
            val_risk_scores.extend(risk_scores.cpu().numpy())
            val_durations.extend(durations.numpy())
            val_events.extend(events.numpy())
    
    # Compute C-index
    val_risk_scores = np.array(val_risk_scores)
    val_durations = np.array(val_durations)
    val_events = np.array(val_events)
    
    c_index = concordance_index(val_durations, val_risk_scores, val_events)
    
    # Save best model
    model_path = get_model_path(scenario_name)
    
    if os.path.exists(model_path):
        # Load existing model and compare
        existing_model = torch.load(model_path, map_location=device)
        val_scores_existing = score_model_test(existing_model, df_test, scenario_name, device)
        
        if c_index > val_scores_existing:
            print(f"New model C-index: {c_index:.4f} > Existing: {val_scores_existing:.4f}")
            torch.save(model, model_path)
        else:
            print(f"Keeping existing model. Current: {c_index:.4f} < Existing: {val_scores_existing:.4f}")
    else:
        print(f"Saving initial model with C-index: {c_index:.4f}")
        torch.save(model, model_path)
    
    trial.set_user_attr(key="model", value=model)
    
    return c_index


def score_model_test(model: SurvTimeSurvival, df_test, scenario_name: ExperimentScenario, device):
    """Score model on test data"""
    test_dataset = SurvTimeSurvivalDataset(df_test, scenario_name)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
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
            
            # Get risk scores using the model's prediction method
            risk_scores = model.predict_risk_score(input_nums=input_nums, seq_lengths=seq_lengths)
            
            test_risk_scores.extend(risk_scores.cpu().numpy())
            test_durations.extend(durations.numpy())
            test_events.extend(events.numpy())
    
    test_risk_scores = np.array(test_risk_scores)
    test_durations = np.array(test_durations)
    test_events = np.array(test_events)
    
    return concordance_index(test_durations, test_risk_scores, test_events)


def get_device():
    n_cuda = np.random.randint(0, 4)
    return torch.device("cuda:"+ str(n_cuda) if torch.cuda.is_available() else "cpu")


def get_model_path(scenario_name: ExperimentScenario):
    """Get model path based on scenario"""
    if scenario_name == ExperimentScenario.NON_TIME_VARIANT:
        return egfr_ti_survtimesurvival_model_path
    elif scenario_name == ExperimentScenario.TIME_VARIANT:
        return egfr_tv_survtimesurvival_model_path
    elif scenario_name == ExperimentScenario.HETEROGENEOUS:
        return hg_survtimesurvival_model_path
    elif scenario_name == ExperimentScenario.EGFR_COMPONENTS:
        return egfr_components_survtimesurvival_model_path


def run(scenario_name: ExperimentScenario):
    print(f"Running SurvTimeSurvival experiment for {scenario_name.value}")
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    df, df_test = get_train_test_data(scenario_name)
    print(f"Training samples: {len(df)}, Test samples: {len(df_test)}")
    
    # Hyperparameter optimization
    def objective_wrapper(trial):
        return objective(trial, scenario_name)
    
    best_model = ex_optuna(objective_wrapper)
    
    # Load best model for final evaluation
    model_path = get_model_path(scenario_name)
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location=device)
    else:
        model = best_model
        
    model.eval()
    
    # Final evaluation
    test_dataset = SurvTimeSurvivalDataset(df_test, scenario_name)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    test_risk_scores = []
    test_durations = []
    test_events = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_nums = batch['input_nums'].to(device)
            durations = batch['duration']
            events = batch['event']
            seq_lengths = batch['seq_length'].to(device)
            
            # Get risk scores using the model's prediction method
            risk_scores = model.predict_risk_score(input_nums=input_nums, seq_lengths=seq_lengths)
            
            test_risk_scores.extend(risk_scores.cpu().numpy())
            test_durations.extend(durations.numpy())
            test_events.extend(events.numpy())
    
    test_risk_scores = np.array(test_risk_scores)
    test_durations = np.array(test_durations)
    test_events = np.array(test_events)
    
    # Compute metrics
    print("\n=== Final Results ===")
    
    # C-Index
    c_index = concordance_index(test_durations, test_risk_scores, test_events)
    print(f"C-Index on Test Data: {round_metric(c_index)}")
    
    # Brier Score
    brier_score = compute_brier_score_from_risk_scores(df, df_test, test_risk_scores)
    if brier_score is not None:
        print(f'Integrated Brier Score Test: {brier_score}')
    
    # Time-dependent AUC
    times = np.arange(1, 365, 1)
    y_train = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df)
    y_test = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df_test)
    
    _, mean_auc = cumulative_dynamic_auc(y_train, y_test, test_risk_scores, times)
    print(f"Mean time-dependent AUC: {round_metric(mean_auc)}")


def run_all_scenarios():
    """Run SurvTimeSurvival experiments for all scenarios"""
    scenarios = [
        ExperimentScenario.NON_TIME_VARIANT,
        ExperimentScenario.TIME_VARIANT,
        ExperimentScenario.HETEROGENEOUS,
        ExperimentScenario.EGFR_COMPONENTS
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"Running SurvTimeSurvival for {scenario.value}")
        print(f"{'='*50}")
        try:
            run(scenario)
        except Exception as e:
            print(f"Error in {scenario.value}: {str(e)}")
            continue


if __name__ == '__main__':
    run_all_scenarios()
