import torch
import torch.nn as nn
import torch.optim as optim

from pkgs.models.deepsurv import DeepSurv
from pkgs.data.model_data_store import get_train_test_data_egfr
from pkgs.playground.exp_common import batch_size
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
from pkgs.experiments.utils import report_metric

class DeepSurvDataset(Dataset):
    def __init__(self, df, features, duration_col, event_col):
        """
        Args:
            df (pd.DataFrame): DataFrame containing survival data.
            features (list of str): List of feature column names.
            duration_col (str): Column name for survival durations.
            event_col (str): Column name for event indicators.
        """
        self.X = torch.tensor(df[features].values, dtype=torch.float32)
        self.durations = torch.tensor(df[duration_col].values, dtype=torch.float32)
        self.events = torch.tensor(df[event_col].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.durations[idx], self.events[idx]

def neg_log_partial_likelihood(risk, durations, events):
    """
    Computes the negative log partial likelihood for Cox Proportional Hazards model.

    Args:
        risk (Tensor): Predicted risk scores (log hazards), shape [N, 1].
        durations (Tensor): Survival durations, shape [N].
        events (Tensor): Event indicators, shape [N] (1 if event occurred, 0 otherwise).

    Returns:
        loss (Tensor): Scalar loss value.
    """
    risk = risk.view(-1) # Ensure risk is a 1D tensor
    # Sort data by descending durations
    durations_sorted, indices = torch.sort(durations, descending=True)
    risk_sorted = risk[indices]
    events_sorted = events[indices]

    loss = 0.0
    for i in range(len(durations_sorted)):
        event_i = events_sorted[i]
        if event_i == 1: # only calculate loss for observed events
            risk_set = risk_sorted[i:] # Risk set includes all subjects with duration >= current duration
            log_sum_risk = torch.logsumexp(risk_set, dim=0) # Numerically stable log(sum(exp(risk_set)))
            loss -= (risk_sorted[i] - log_sum_risk) # Add negative log likelihood for this event

    return loss

def run():
    features = ['duration_in_days', 'egfr']
    duration_col = 'duration_in_days'
    event_col = 'has_esrd'

    df, df_test = get_train_test_data_egfr(True)

    # Prepare the training dataset and use full-batch training for proper risk set computation.
    train_dataset = DeepSurvDataset(df, features, duration_col, event_col)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    # Hyperparameters
    input_dim = 2
    hidden_dims = [128, 64, 16]
    learning_rate = 0.001
    num_epochs = 25

    # Initialize model and optimizer
    model = DeepSurv(input_dim, hidden_dims)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            X_batch, durations_batch, events_batch = batch
            optimizer.zero_grad()
            risk_scores = model(X_batch)

            loss = neg_log_partial_likelihood(risk_scores, durations_batch, events_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Evaluate on test data
    # We assume test data only contains features; the survival outcomes are unknown.
    X_test = torch.tensor(df_test[features].values, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        test_risk_scores = model(X_test)
        print("Test risk scores shape:", test_risk_scores.shape)

    # Compute the concordance index (using -risk because higher risk implies lower survival)
    c_index = report_metric(concordance_index(df_test['duration_in_days'], test_risk_scores, df_test['has_esrd']))
    print("C-Index on Test Data:", c_index)

if __name__ == '__main__':
    run()