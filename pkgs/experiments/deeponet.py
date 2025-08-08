import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

from pkgs.models.deeponet import DeepONet
from pkgs.data.model_data_store import get_train_test_data
from pkgs.experiments.utils import round_metric, ex_optuna, get_tv_rnn_model_features, compute_brier_score_from_risk_scores
from pkgs.commons import (
    egfr_tv_deeponet_model_path, hg_deeponet_model_path, 
    egfr_components_deeponet_model_path, egfr_ti_deeponet_model_path
)
from pkgs.data.types import ExperimentScenario
import os


class DeepONetDataset(Dataset):
    def __init__(self, df, scenario_name: ExperimentScenario, max_seq_length=50):
        self.df = df
        self.scenario_name = scenario_name
        self.max_seq_length = max_seq_length
        
        # Get features based on scenario
        self.features = get_tv_rnn_model_features(scenario_name)
        if self.features is None:
            # Handle NON_TIME_VARIANT case which isn't covered by get_tv_rnn_model_features
            if scenario_name == ExperimentScenario.NON_TIME_VARIANT:
                self.features = ['egfr']  # Only egfr for non-time-variant scenario
            else:
                raise ValueError(f"No features defined for scenario: {scenario_name}")
        
        # Group by subject_id to create sequences
        self.subjects = df['subject_id'].unique()
        
        # Precompute some statistics for time normalization
        self.max_duration = df['duration_in_days'].max()
        
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        subject_data = self.df[self.df['subject_id'] == subject_id].sort_values('duration_in_days')
        
        # Extract features for this subject (covariate histories)
        if self.scenario_name == ExperimentScenario.NON_TIME_VARIANT:
            # For time-invariant, use the last observation but repeat for sequence
            features = subject_data[self.features].iloc[-1:].values
            features = np.tile(features, (self.max_seq_length, 1))
            seq_length = 1
        else:
            # For time-varying scenarios, use all observations
            features = subject_data[self.features].values
            seq_length = len(features)
            
            # Pad or truncate to max_seq_length
            if seq_length > self.max_seq_length:
                features = features[-self.max_seq_length:]  # Keep last observations
                seq_length = self.max_seq_length
            elif seq_length < self.max_seq_length:
                # Pad with last observation (forward fill)
                last_obs = features[-1:] if len(features) > 0 else np.zeros((1, features.shape[1]))
                padding = np.tile(last_obs, (self.max_seq_length - seq_length, 1))
                features = np.vstack([features, padding])
        
        # Query time point (normalized to [0,1])
        duration = subject_data['duration_in_days'].iloc[-1]
        query_time = duration / self.max_duration  # Normalize to [0,1]
        
        # Event indicator
        event = subject_data['has_esrd'].iloc[-1]
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),  # (seq_len, input_dim)
            'query_time': torch.tensor([[query_time]], dtype=torch.float32),  # (1, 1)
            'duration': torch.tensor(duration, dtype=torch.float32),
            'event': torch.tensor(event, dtype=torch.float32),
            'seq_length': torch.tensor(seq_length, dtype=torch.long)
        }


def deeponet_survival_loss(model, u, query_times, durations, events):
    """
    Custom survival loss for DeepONet based on likelihood for censored data
    """
    batch_size = u.size(0)
    device = u.device
    
    # Get risk scores at each patient's event/censoring time
    risk_scores = []
    
    for i in range(batch_size):
        u_i = u[i:i+1]  # (1, seq_len, input_dim)
        q_i = query_times[i:i+1]  # (1, 1)
        
        score = model.forward(u_i, q_i)  # (1, 1)
        risk_scores.append(score.squeeze())
    
    risk_scores = torch.stack(risk_scores)  # (batch_size,)
    
    # Partial likelihood (Cox-style loss)
    risk_exp = torch.exp(risk_scores)
    
    loss = 0.0
    num_events = 0
    
    for i in range(batch_size):
        if events[i] == 1:  # Event occurred
            # Risk set: all patients with duration >= current patient's duration
            at_risk = durations >= durations[i]
            risk_set_sum = torch.sum(risk_exp[at_risk])
            
            if risk_set_sum > 0:
                loss += risk_scores[i] - torch.log(risk_set_sum)
                num_events += 1
    
    if num_events > 0:
        loss = -loss / num_events
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    return loss


def objective(trial, scenario_name: ExperimentScenario):
    # Hyperparameter suggestions
    branch_hidden_dims = trial.suggest_categorical('branch_hidden_dims', 
                                                  [(64, 128), (128, 256), (256, 512)])
    trunk_hidden_dims = trial.suggest_categorical('trunk_hidden_dims',
                                                 [(64, 128), (128, 256), (256, 512)])
    operator_dim = trial.suggest_categorical('operator_dim', [64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    device = get_device()
    
    # Load data
    df, df_test = get_train_test_data(scenario_name)
    
    # Create datasets
    train_dataset = DeepONetDataset(df, scenario_name)
    val_dataset = DeepONetDataset(df_test, scenario_name)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model parameters
    features = get_tv_rnn_model_features(scenario_name)
    input_dim = len(features)
    
    # Initialize model
    model = DeepONet(
        input_dim=input_dim,
        branch_hidden_dims=branch_hidden_dims,
        trunk_hidden_dims=trunk_hidden_dims,
        query_dim=1,
        dropout=dropout,
        operator_dim=operator_dim
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
            features = batch['features'].to(device)  # (batch_size, seq_len, input_dim)
            query_times = batch['query_time'].to(device)  # (batch_size, 1, 1)
            durations = batch['duration'].to(device)
            events = batch['event'].to(device)
            
            # Reshape query_times for model
            query_times = query_times.squeeze(1)  # (batch_size, 1)
            
            optimizer.zero_grad()
            
            # Compute loss using custom survival loss
            loss = deeponet_survival_loss(model, features, query_times, durations, events)
            
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0:
            epoch_loss /= num_batches
            
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
    
    # Validation
    model.eval()
    val_risk_scores = []
    val_durations = []
    val_events = []
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            query_times = batch['query_time'].to(device)
            durations = batch['duration']
            events = batch['event']
            
            query_times = query_times.squeeze(1)  # (batch_size, 1)
            
            # Get risk scores
            risk_scores = model.predict_risk_score(features)
            
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


def score_model_test(model: DeepONet, df_test, scenario_name: ExperimentScenario, device):
    """Score model on test data"""
    test_dataset = DeepONetDataset(df_test, scenario_name)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    test_risk_scores = []
    test_durations = []
    test_events = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            durations = batch['duration']
            events = batch['event']
            
            # Get risk scores
            risk_scores = model.predict_risk_score(features)
            
            test_risk_scores.extend(risk_scores.cpu().numpy())
            test_durations.extend(durations.numpy())
            test_events.extend(events.numpy())
    
    test_risk_scores = np.array(test_risk_scores)
    test_durations = np.array(test_durations)
    test_events = np.array(test_events)
    
    return concordance_index(test_durations, test_risk_scores, test_events)


def get_device():
    return torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def get_model_path(scenario_name: ExperimentScenario):
    """Get model path based on scenario"""
    if scenario_name == ExperimentScenario.NON_TIME_VARIANT:
        return egfr_ti_deeponet_model_path
    elif scenario_name == ExperimentScenario.TIME_VARIANT:
        return egfr_tv_deeponet_model_path
    elif scenario_name == ExperimentScenario.HETEROGENEOUS:
        return hg_deeponet_model_path
    elif scenario_name == ExperimentScenario.EGFR_COMPONENTS:
        return egfr_components_deeponet_model_path


def run(scenario_name: ExperimentScenario):
    print(f"Running DeepONet experiment for {scenario_name.value}")
    
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
    test_dataset = DeepONetDataset(df_test, scenario_name)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    test_risk_scores = []
    test_durations = []
    test_events = []
    test_survival_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            durations = batch['duration']
            events = batch['event']
            
            # Get risk scores
            risk_scores = model.predict_risk_score(features)
            
            # Get survival probabilities at various time points
            time_points = torch.linspace(0.1, 1.0, 10, device=device).unsqueeze(-1)  # (10, 1)
            survival_probs = model.predict_survival(features, time_points)  # (batch_size, 10)
            
            test_risk_scores.extend(risk_scores.cpu().numpy())
            test_durations.extend(durations.numpy())
            test_events.extend(events.numpy())
            test_survival_probs.extend(survival_probs.cpu().numpy())
    
    test_risk_scores = np.array(test_risk_scores)
    test_durations = np.array(test_durations)
    test_events = np.array(test_events)
    test_survival_probs = np.array(test_survival_probs)
    
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
    
    # Additional DeepONet-specific evaluation
    print("\n=== DeepONet Specific Evaluation ===")
    
    # Evaluate survival function predictions at different time points
    with torch.no_grad():
        # Take a sample of test data for detailed evaluation
        sample_indices = np.random.choice(len(test_dataset), min(100, len(test_dataset)), replace=False)
        sample_data = [test_dataset[i] for i in sample_indices]
        
        sample_features = torch.stack([item['features'] for item in sample_data]).to(device)
        sample_durations = np.array([item['duration'].item() for item in sample_data])
        sample_events = np.array([item['event'].item() for item in sample_data])
        
        # Evaluate at multiple time points (normalized)
        max_duration = df['duration_in_days'].max()
        eval_times = torch.tensor([[t/max_duration] for t in [30, 90, 180, 365]], device=device)  # (4, 1)
        
        survival_preds = model.predict_survival(sample_features, eval_times)  # (sample_size, 4)
        
        print(f"Average survival probabilities at different time points:")
        time_labels = ["30 days", "90 days", "180 days", "365 days"]
        for i, label in enumerate(time_labels):
            avg_surv = survival_preds[:, i].mean().item()
            print(f"  {label}: {avg_surv:.3f}")


def run_all_scenarios():
    """Run DeepONet experiments for all scenarios"""
    scenarios = [
        ExperimentScenario.NON_TIME_VARIANT,
        ExperimentScenario.TIME_VARIANT,
        ExperimentScenario.HETEROGENEOUS,
        ExperimentScenario.EGFR_COMPONENTS
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"Running DeepONet for {scenario.value}")
        print(f"{'='*50}")
        try:
            run(scenario)
        except Exception as e:
            print(f"Error in {scenario.value}: {str(e)}")
            continue


if __name__ == '__main__':
    run_all_scenarios()
