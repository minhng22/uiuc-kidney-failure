import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index

def generate_sample_data(num_subjects=100, max_observations=30, seed=42):
    """
    Generate sample longitudinal data with time-varying covariates.
    """
    np.random.seed(seed)
    data = []
    
    for subject in range(num_subjects):
        # Initialize subject-specific parameters
        baseline_egfr = np.random.normal(75, 15)  # Baseline eGFR
        egfr_slope = np.random.normal(-0.1, 0.05)  # Individual decline rate
        observation_points = np.random.randint(10, max_observations)
        
        # Generate observation times (not necessarily equally spaced)
        times = np.sort(np.random.choice(range(max_observations * 30), 
                                       size=observation_points, 
                                       replace=False))
        
        # Risk of events increases as eGFR decreases
        has_esrd = False
        is_dead = False
        
        for i, t in enumerate(times):
            # Calculate current eGFR with some random noise
            current_egfr = baseline_egfr + egfr_slope * t + np.random.normal(0, 2)
            
            # Event probabilities based on current eGFR
            esrd_prob = 1 / (1 + np.exp(0.1 * (current_egfr - 15)))  # Higher risk when eGFR < 15
            death_prob = 1 / (1 + np.exp(0.05 * (current_egfr - 30)))  # Higher risk when eGFR < 30
            
            # Determine if events occur
            if not has_esrd and not is_dead:
                has_esrd = np.random.random() < esrd_prob
                if not has_esrd:
                    is_dead = np.random.random() < death_prob
            
            data.append({
                'subject_id': subject,
                'duration_in_days': t,
                'start': times[i-1] if i > 0 else 0,
                'stop': t,
                'has_esrd': 1e-5 if not has_esrd else 1,
                'dead': 1 if is_dead else 0,
                'egfr': current_egfr
            })
            
            if has_esrd or is_dead:
                break
                
    return pd.DataFrame(data)

def calculate_c_index(hazard_preds, time_intervals, event_indicators, num_risks):
    """
    Calculate the concordance index (C-index) for survival predictions.
    """
    batch_size = hazard_preds.size(0)
    c_index_per_risk = []

    for risk in range(num_risks):
        risk_hazard_preds = hazard_preds[:, risk, :]
        risk_event_indicators = event_indicators[:, risk]
        observed_times = time_intervals[:, 0]

        # Compute risk scores as the cumulative hazard
        risk_scores = -torch.sum(torch.log(1 - risk_hazard_preds + 1e-8), dim=1).detach().cpu().numpy()

        # Only include cases where an event occurred for the risk
        mask = risk_event_indicators > 0
        c_index = concordance_index(
            observed_times[mask].detach().cpu().numpy(),
            risk_scores[mask],
            event_observed=risk_event_indicators[mask].detach().cpu().numpy()
        )
        c_index_per_risk.append(c_index)

    return c_index_per_risk

class LongitudinalDataset(Dataset):
    """Dataset for handling longitudinal data with time-varying covariates"""
    def __init__(self, df, max_seq_length=None):
        self.subject_groups = list(df.groupby('subject_id'))
        self.max_seq_length = max_seq_length or max(df.groupby('subject_id').size())
        
        # Normalize numerical features
        self.egfr_mean = df['egfr'].mean()
        self.egfr_std = df['egfr'].std()
        
    def __len__(self):
        return len(self.subject_groups)
    
    def __getitem__(self, idx):
        subject_id, subject_data = self.subject_groups[idx]
        seq_length = len(subject_data)
        
        # Create feature matrix
        features = np.zeros((self.max_seq_length, 3))  # [start, stop, egfr]
        mask = np.zeros(self.max_seq_length)
        
        # Fill in features
        features[:seq_length, 0] = subject_data['start'].values
        features[:seq_length, 1] = subject_data['stop'].values
        features[:seq_length, 2] = (subject_data['egfr'].values - self.egfr_mean) / self.egfr_std
        
        # Create mask for valid timesteps
        mask[:seq_length] = 1
        
        # Get time to event and event indicators
        time_to_event = subject_data['duration_in_days'].iloc[-1]
        events = np.array([
            subject_data['has_esrd'].iloc[-1],
            subject_data['dead'].iloc[-1]
        ])
        
        return (torch.FloatTensor(features),
                torch.FloatTensor(mask),
                torch.LongTensor([time_to_event]),
                torch.FloatTensor(events))

def survival_loss(hazard_preds, time_intervals, event_indicators, num_risks):
    """
    Compute the loss for competing risks in the survival setting.
    """
    batch_size = hazard_preds.size(0)
    loss = 0

    for risk in range(num_risks):
        # Extract relevant hazard predictions for the current risk
        risk_hazard_preds = hazard_preds[:, risk, :]
        risk_event_indicators = event_indicators[:, risk]

        # Event log probability
        time_indices = time_intervals[:, 0].clamp(max=risk_hazard_preds.size(1) - 1).long()
        event_log_prob = torch.log(risk_hazard_preds[torch.arange(batch_size), time_indices]) * risk_event_indicators

        # Censoring log probability
        censor_log_prob = torch.zeros(batch_size, device=risk_hazard_preds.device)
        for i in range(batch_size):
            t = time_indices[i].item()  # Ensure t is scalar
            censor_log_prob[i] = torch.sum(torch.log(1 - risk_hazard_preds[i, :t + 1]))

        censor_log_prob = censor_log_prob * (1 - risk_event_indicators)

        # Add to total loss
        loss += -torch.mean(event_log_prob + censor_log_prob)

    return loss

# Hyperparameters
input_dim = 3  # Start, stop, and normalized eGFR
hidden_dims = [64, 32]
num_risks = 2
time_bins = 30
batch_size = 16
learning_rate = 1e-3
num_epochs = 50
