import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# Dynamic DeepHit Model
class DynamicDeepHit(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_risks, time_bins):
        """
        Dynamic DeepHit model for competing risks.
        Args:
            input_dim: Number of input features (e.g., egfr, start, stop, etc.).
            hidden_dim: Number of hidden units in the dense layers.
            num_risks: Number of competing risks (e.g., has_esrd, dead).
            time_bins: Number of discrete time bins.
        """
        super(DynamicDeepHit, self).__init__()
        self.num_risks = num_risks
        self.time_bins = time_bins

        # Shared fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Separate output layers for each risk
        self.risk_heads = nn.ModuleList([nn.Linear(hidden_dim, time_bins) for _ in range(num_risks)])

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x: Input tensor of shape (batch_size, time_steps, input_dim).
        Returns:
            hazard_preds: Predicted hazard functions for each risk (batch_size, num_risks, time_bins).
        """
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        # Compute hazard predictions for each risk
        hazard_preds = torch.stack([torch.sigmoid(risk_head(h)) for risk_head in self.risk_heads], dim=1)
        return hazard_preds

# Survival Loss Function
def survival_loss(hazard_preds, time_intervals, event_indicators, num_risks):
    """
    Compute the loss for competing risks in the survival setting.
    Args:
        hazard_preds: Predicted hazard functions (batch_size, num_risks, time_bins).
        time_intervals: Observed time intervals (batch_size,).
        event_indicators: Event indicators for each risk (batch_size, num_risks).
        num_risks: Number of competing risks.
    Returns:
        Loss value (scalar).
    """
    batch_size = hazard_preds.size(0)
    loss = 0

    for risk in range(num_risks):
        # Extract relevant hazard predictions for the current risk
        risk_hazard_preds = hazard_preds[:, risk, :]
        risk_event_indicators = event_indicators[:, risk]

        # Event log probability
        event_log_prob = torch.log(risk_hazard_preds[torch.arange(batch_size), time_intervals]) * risk_event_indicators

        # Censoring log probability
        censor_log_prob = torch.sum(torch.log(1 - risk_hazard_preds[:, :time_intervals + 1]), dim=1) * (1 - risk_event_indicators)

        # Add to total loss
        loss += -torch.mean(event_log_prob + censor_log_prob)

    return loss

# Concordance Index (C-Index) Calculation
def c_index(hazard_preds, time_intervals, event_indicators):
    """
    Calculate the concordance index (C-Index) for survival predictions.
    Args:
        hazard_preds: Predicted hazard functions (batch_size, num_risks, time_bins).
        time_intervals: Observed time intervals (batch_size,).
        event_indicators: Event indicators for each risk (batch_size, num_risks).
    Returns:
        C-Index value (scalar).
    """
    batch_size = hazard_preds.size(0)
    concordant = 0
    permissible = 0

    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if time_intervals[i] != time_intervals[j]:
                permissible += 1
                risk_i = torch.sum(hazard_preds[i, :, :time_intervals[i] + 1])
                risk_j = torch.sum(hazard_preds[j, :, :time_intervals[j] + 1])

                if (time_intervals[i] < time_intervals[j] and risk_i > risk_j) or \
                   (time_intervals[i] > time_intervals[j] and risk_i < risk_j):
                    concordant += 1

    return concordant / permissible if permissible > 0 else 0.0

import random
import pandas as pd

# Generate synthetic data
num_subjects = 100
time_bins = 100
data = []

for subject_id in range(num_subjects):
    start_time = 0.0
    stop_time = random.uniform(0, time_bins)
    duration_in_days = random.randint(0, time_bins)
    has_esrd = random.choice([0, 1])
    dead = random.choice([0, 1]) if not has_esrd else 0  # Mutually exclusive risks
    egfr = random.uniform(60, 120)  # Example range for eGFR

    data.append({
        "subject_id": subject_id,
        "duration_in_days": duration_in_days,
        "start": start_time,
        "stop": stop_time,
        "has_esrd": has_esrd,
        "dead": dead,
        "egfr": egfr
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert to PyTorch tensors
features = torch.tensor(df[["start", "stop", "egfr"]].values, dtype=torch.float32)  # Dynamic features
time_intervals = torch.tensor(df["duration_in_days"].values, dtype=torch.long)  # Observed times
event_indicators = torch.tensor(df[["has_esrd", "dead"]].values, dtype=torch.float32)  # Competing risks

# Reshape for batch processing
batch_size = len(df["subject_id"].unique())
features = features.view(batch_size, -1, features.size(1))
time_intervals = time_intervals.view(batch_size, -1)
event_indicators = event_indicators.view(batch_size, -1, 2)

# Initialize the Model
input_dim = 3  # Number of dynamic features (e.g., start, stop, egfr)
hidden_dim = 128
num_risks = 2  # Competing risks (has_esrd, dead)
time_bins = 100

model = DynamicDeepHit(input_dim, hidden_dim, num_risks, time_bins)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 10
for epoch in range(epochs):
    hazard_preds = model(features)
    loss = survival_loss(hazard_preds, time_intervals, event_indicators, num_risks)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate C-Index
    cidx = c_index(hazard_preds, time_intervals, event_indicators)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, C-Index: {cidx:.4f}")