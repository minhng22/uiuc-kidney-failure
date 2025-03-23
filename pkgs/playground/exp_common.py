import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def generate_sample_data(num_subjects=100, max_observations=30, seed=42):
    """
    Generate sample longitudinal data with time-varying covariates.
    """
    np.random.seed(seed)
    data = []
    
    for subject in range(num_subjects):
        baseline_egfr = np.random.normal(75, 15)
        egfr_slope = np.random.normal(-0.1, 0.05)
        observation_points = np.random.randint(10, max_observations)
        times = np.sort(np.random.choice(range(max_observations * 30), size=observation_points, replace=False))
        
        has_esrd = False
        is_dead = False
        
        for i, t in enumerate(times):
            current_egfr = baseline_egfr + egfr_slope * t + np.random.normal(0, 2)
            
            has_esrd = np.random.choice([0, 1])
            is_dead = np.random.choice([0, 1])
            
            data.append({
                'subject_id': subject,
                'duration_in_days': t,
                'start': times[i - 1] if i > 0 else 0,
                'stop': t,
                'has_esrd': has_esrd,
                'dead': is_dead,
                'egfr': current_egfr
            })
            
            if is_dead:
                break

    # Ensure there is a mix of censored and uncensored subjects
    if all(d['has_esrd'] == 0 for d in data):
        # Force at least one subject to experience the event
        random_subject = np.random.choice(data)['subject_id']
        for record in data:
            if record['subject_id'] == random_subject:
                record['has_esrd'] = 1
                break

    return pd.DataFrame(data)

def calculate_c_index(hazard_preds, time_intervals, event_indicators, num_risks):
    """
    Calculate the concordance index (C-index) for survival predictions.
    """
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

# Dataset that supports RNN and attention models
class RNNAttentionDataset(Dataset):
    """Dataset for handling longitudinal data with time-varying covariates"""
    def __init__(self, df, max_seq_length=None, multiple_risk=True):
        self.subject_groups = list(df.groupby('subject_id'))
        self.max_seq_length = max_seq_length or max(df.groupby('subject_id').size())
        
        # Normalize numerical features
        self.egfr_mean = df['egfr'].mean()
        self.egfr_std = df['egfr'].std()
        self.multiple_risk = multiple_risk
        
    def __len__(self):
        return len(self.subject_groups)
    
    def __getitem__(self, idx):
        _, subject_data = self.subject_groups[idx]
        seq_length = len(subject_data)
        
        # Create feature matrix
        features = np.zeros((self.max_seq_length, 2))  # ['duration_in_days', 'egfr']. 'duration_in_days' is better than 'start', 'stop' since admission time can be vastly different between patients.
        mask = np.zeros(self.max_seq_length)
        
        # Fill in features
        features[:seq_length, 0] = subject_data['duration_in_days'].values
        features[:seq_length, 1] = (subject_data['egfr'].values - self.egfr_mean) / self.egfr_std
        
        # Create mask for valid timesteps
        mask[:seq_length] = 1
        
        # Get time to event and event indicators
        time_to_event = subject_data['duration_in_days'].iloc[-1]
        if self.multiple_risk:
            events = np.array([
                subject_data['has_esrd'].iloc[-1],
                subject_data['dead'].iloc[-1]
            ])
        else:
            events = np.array([
                subject_data['has_esrd'].iloc[-1],
            ])
        
        return (torch.FloatTensor(features),
                torch.FloatTensor(mask),
                torch.LongTensor([time_to_event]),
                torch.FloatTensor(events))

def survival_loss(hazard_preds, time_intervals, event_indicators, num_risks, alpha=0.5, sigma=0.1):
    """
    Compute the Dynamic-DeepHit loss for competing risks in the survival setting.

    Args:
        alpha (float): Weight for the ranking loss component.
        sigma (float): Parameter for the ranking loss function.
    """
    batch_size = hazard_preds.size(0)
    num_timepoints = hazard_preds.size(2)

    # Initialize total loss
    total_loss = 0

    for risk in range(num_risks):
        # Extract relevant hazard predictions and event indicators for the current risk
        risk_hazard_preds = hazard_preds[:, risk, :]
        risk_event_indicators = event_indicators[:, risk]

        # Ensure time indices are within valid range
        time_indices = time_intervals[:, 0].clamp(max=num_timepoints - 1).long()

        # Log-likelihood loss
        event_log_prob = torch.log(risk_hazard_preds[torch.arange(batch_size), time_indices]) * risk_event_indicators

        censor_log_prob = torch.zeros(batch_size, device=risk_hazard_preds.device)
        for i in range(batch_size):
            t = time_indices[i].item()
            if t > 0:
                censor_log_prob[i] = torch.sum(torch.log(1 - risk_hazard_preds[i, :t]))

        censor_log_prob = censor_log_prob * (1 - risk_event_indicators)

        log_likelihood_loss = -torch.mean(event_log_prob + censor_log_prob)

        # Ranking loss
        ranking_loss = 0
        count = 0
        for i in range(batch_size):
            for j in range(batch_size):
                if time_intervals[i] < time_intervals[j] and risk_event_indicators[i] == 1:
                    t_i = time_indices[i].item()
                    F_i = torch.sum(risk_hazard_preds[i, :t_i])
                    F_j = torch.sum(risk_hazard_preds[j, :t_i])
                    ranking_loss += torch.exp(-(F_i - F_j) / sigma)
                    count += 1

        if count > 0:
            ranking_loss /= count

        # Combine log-likelihood and ranking loss
        total_loss += log_likelihood_loss + alpha * ranking_loss

    return total_loss / num_risks

# Hyperparameters
input_dim = 2 #['duration_in_days', 'egfr']
hidden_dims = [64, 32]
num_risks_multiple_risks = 2
time_bins = 300
batch_size = 16
learning_rate = 1e-3
num_epochs = 1
