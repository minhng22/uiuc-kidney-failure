import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicDeepHit(nn.Module):
    def __init__(self, input_static_dim, input_dynamic_dim, hidden_dim, output_dim, time_bins, dropout=0.3):
        """
        Initializes the Dynamic DeepHit model.
        Args:
            input_static_dim: Dimension of static covariates.
            input_dynamic_dim: Dimension of dynamic covariates.
            hidden_dim: Number of hidden units in shared layers.
            output_dim: Output dimension for each time bin (number of risk categories, usually 1 for binary).
            time_bins: Number of discrete time bins for the survival prediction.
            dropout: Dropout rate for regularization.
        """
        super(DynamicDeepHit, self).__init__()

        # Shared layers for static features
        self.static_fc = nn.Sequential(
            nn.Linear(input_static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Shared layers for dynamic features
        self.dynamic_fc = nn.Sequential(
            nn.Linear(input_dynamic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Shared hidden layers
        self.shared_fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Output layers for each time bin
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(time_bins)
        ])

    def forward(self, static_covariates, dynamic_covariates):
        """
        Forward pass for the model.
        Args:
            static_covariates: Static features (batch_size, input_static_dim).
            dynamic_covariates: Dynamic features (batch_size, input_dynamic_dim).
        Returns:
            Predicted hazard functions for each time bin (batch_size, time_bins, output_dim).
        """
        static_features = self.static_fc(static_covariates)
        dynamic_features = self.dynamic_fc(dynamic_covariates)
        shared_features = torch.cat([static_features, dynamic_features], dim=1)
        shared_output = self.shared_fc(shared_features)

        # Compute hazard predictions for each time bin
        hazard_predictions = torch.stack([layer(shared_output) for layer in self.output_layers], dim=1)

        # Apply softmax across time bins for interpretability
        return F.softmax(hazard_predictions, dim=1)

def survival_loss(hazard_preds, time_intervals, event_indicators):
    """
    Negative log-likelihood for survival data.
    Args:
        hazard_preds: Predicted hazard functions (batch_size, time_bins, output_dim).
        time_intervals: Observed time intervals (batch_size,).
        event_indicators: Event indicators (1 for event, 0 for censoring) (batch_size,).
    Returns:
        Loss value (scalar).
    """
    batch_size, time_bins, _ = hazard_preds.size()
    
    # Select hazard predictions for observed time intervals
    event_log_prob = torch.log(hazard_preds[torch.arange(batch_size), time_intervals.squeeze()]) * event_indicators

    # Compute cumulative hazard up to the censoring or event time
    censor_log_prob = torch.zeros(batch_size, device=hazard_preds.device)
    for i in range(batch_size):
        censor_log_prob[i] = torch.sum(hazard_preds[i, :time_intervals[i] + 1]) * (1 - event_indicators[i])

    loss = -torch.mean(event_log_prob + torch.log(censor_log_prob + 1e-8))
    return loss

def concordance_index(hazard_preds, time_intervals, event_indicators):
    """
    Computes the concordance index (c-index) for survival predictions.
    Args:
        hazard_preds: Predicted hazard functions (batch_size, time_bins, output_dim).
        time_intervals: Observed time intervals (batch_size,).
        event_indicators: Event indicators (1 for event, 0 for censoring) (batch_size,).
    Returns:
        c_index: Concordance index (scalar).
    """
    batch_size = hazard_preds.size(0)
    survival_scores = -torch.cumsum(hazard_preds, dim=1)  # Calculate survival scores as negative cumulative hazard

    # Concordance calculation
    num_correct = 0
    num_possible = 0

    for i in range(batch_size):
        for j in range(batch_size):
            if i != j:
                # Define comparable pairs (event occurred and survival time is known)
                if event_indicators[i] == 1 and time_intervals[i] < time_intervals[j]:
                    num_possible += 1
                    if survival_scores[i, time_intervals[i]] > survival_scores[j, time_intervals[i]]:
                        num_correct += 1

    # Compute c-index
    c_index = num_correct / num_possible if num_possible > 0 else 0
    return c_index

# Example usage
if __name__ == "__main__":
    # Model parameters
    input_static_dim = 5
    input_dynamic_dim = 10
    hidden_dim = 32
    output_dim = 1
    time_bins = 50
    batch_size = 16

    # Create model
    model = DynamicDeepHit(input_static_dim, input_dynamic_dim, hidden_dim, output_dim, time_bins)

    # Example data
    static_covariates = torch.rand(batch_size, input_static_dim)
    dynamic_covariates = torch.rand(batch_size, input_dynamic_dim)
    time_intervals = torch.randint(0, time_bins, (batch_size,))
    event_indicators = torch.randint(0, 2, (batch_size,))

    # Forward pass
    hazard_preds = model(static_covariates, dynamic_covariates)

    # Compute loss
    loss = survival_loss(hazard_preds, time_intervals, event_indicators)
    print(f"Loss: {loss.item()}")

    # Compute c-index
    c_index = concordance_index(hazard_preds.squeeze(-1), time_intervals, event_indicators)
    print(f"Concordance Index (c-index): {c_index:.4f}")

