import torch
import torch.nn as nn

class RNNSurv(nn.Module):
    """
    RNNSurv model implementation using LSTM.
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        """
        Initializes the RNNSurv model.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Size of the LSTM hidden state.
            num_layers (int, optional): Number of LSTM layers. Defaults to 1.
        """
        super(RNNSurv, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # batch_first=True for input shape (batch, seq_len, features)
        self.fc = nn.Linear(hidden_size, 1) # Output layer to predict risk score

    def forward(self, x):
        """
        Forward pass of the RNNSurv network.

        Args:
            x (torch.Tensor): Input features of shape (batch, seq_len, input_size).
                              For non-sequential data, seq_len can be 1.

        Returns:
            torch.Tensor: Predicted risk scores (hazard scores) of shape (batch, 1).
        """
        # Assuming input x is already in the shape (batch, seq_len, input_size)
        out, _ = self.rnn(x)

        # Use the output of the last time step for prediction (many-to-one RNN)
        # out[:, -1, :] selects the output from the last time step
        risk_scores = self.fc(out[:, -1, :])
        return risk_scores