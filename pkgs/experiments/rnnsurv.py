import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
import pandas as pd
import numpy as np

from pkgs.models.rnnsurv import RNNSurv # Import the RNNSurv model
from pkgs.data.model_data_store import get_train_test_data_egfr
from pkgs.experiments.utils import report_metric

class DeepSurvDataset(Dataset): # Reusing DeepSurvDataset - general enough for survival data
    def __init__(self, df, features, duration_col, event_col):
        """
        Args:
            df (pd.DataFrame): DataFrame containing survival data.
            features (list of str): List of feature column names (sequences).
            duration_col (str): Column name for survival durations.
            event_col (str): Column name for event indicators.
        """
        # Assuming features is a list of column names where each column is a sequence
        # For simplicity, handling only one sequence feature column in this example
        self.X = torch.tensor(np.stack(df[features[0]].values), dtype=torch.float32).unsqueeze(1) # Assuming features[0] is sequence, make it (N, seq_len, feature_dim) - unsqueeze for seq_len=1 if needed
        self.durations = torch.tensor(df[duration_col].values, dtype=torch.float32)
        self.events = torch.tensor(df[event_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.durations[idx], self.events[idx]

def neg_log_partial_likelihood(risk, durations, events):
    """
    Computes the negative log partial likelihood for Cox Proportional Hazards model.
    (Same loss function as in your DeepSurv implementation - can reuse)
    """
    risk = risk.view(-1)
    durations_sorted, indices = torch.sort(durations, descending=True)
    risk_sorted = risk[indices]
    events_sorted = events[indices]

    loss = 0.0
    for i in range(len(durations_sorted)):
        event_i = events_sorted[i]
        if event_i == 1:
            risk_set = risk_sorted[i:]
            log_sum_risk = torch.logsumexp(risk_set, dim=0)
            loss -= (risk_sorted[i] - log_sum_risk)
    return loss


def run():
    features = ['duration_in_days', 'egfr'] # Now expecting a sequence feature
    duration_col = 'duration_in_days'
    event_col = 'has_esrd'

    df, df_test = get_train_test_data_egfr(True) # Use dummy data with sequence

    # Prepare the training dataset
    train_dataset = DeepSurvDataset(df, features, duration_col, event_col)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True) # Full batch for risk set

    # Hyperparameters for RNNSurv
    input_dim = 2 # Feature dimension of the sequence (assuming egfr_seq is a sequence of 2-dim vectors for example)
    hidden_size = 32
    num_layers = 2
    learning_rate = 0.001
    num_epochs = 25

    # Initialize RNNSurv model and optimizer
    model = RNNSurv(input_dim, hidden_size, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            X_batch, durations_batch, events_batch = batch
            optimizer.zero_grad()
            risk_scores = model(X_batch) # Forward pass with RNN
            loss = neg_log_partial_likelihood(risk_scores, durations_batch, events_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Evaluate on test data
    X_test_seq = torch.tensor(np.stack(df_test[features[0]].values), dtype=torch.float32).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        test_risk_scores = model(X_test_seq)
        print("Test risk scores shape:", test_risk_scores.shape)

    # Compute the concordance index
    c_index_value = concordance_index(df_test['duration_in_days'], -test_risk_scores.numpy(), df_test['has_esrd'])
    c_index = report_metric(c_index_value)
    print("C-Index on Test Data:", c_index)

if __name__ == '__main__':
    run()