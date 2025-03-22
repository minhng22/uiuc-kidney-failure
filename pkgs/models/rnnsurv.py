import torch
import torch.nn as nn

class RNNSurv(nn.Module):
    """
    RNNSurv model implementation based on the provided description.
    """
    def __init__(self, input_size, embedding_size, num_embedding_layers, hidden_size, num_recurrent_layers):
        """
        Initializes the RNNSurv model.

        Args:
            input_size (int): Number of input features (including time interval identifier).
            embedding_size (int): Dimensionality of the embedding layers.
            num_embedding_layers (int): Number of embedding layers (N1).
            hidden_size (int): Size of the LSTM hidden state.
            num_recurrent_layers (int): Number of LSTM layers (N2).
        """
        super(RNNSurv, self).__init__()
        self.embedding_layers = nn.Sequential()
        for i in range(num_embedding_layers):
            input_dim = input_size if i == 0 else embedding_size
            self.embedding_layers.add_module(f'embedding_{i}', nn.Linear(input_dim, embedding_size))
            if i < num_embedding_layers - 1:
                self.embedding_layers.add_module(f'relu_embedding_{i}', nn.ReLU())

        self.rnn = nn.LSTM(embedding_size, hidden_size, num_recurrent_layers, batch_first=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the RNNSurv network.

        Args:
            x (torch.Tensor): Input features of shape (batch, seq_len, input_size),
                                        where input_size includes the features and time interval.

        Returns:
            torch.Tensor: Predicted risk scores (hazard scores) of shape (batch, seq_len, 1) after sigmoid.
        """
        # Pass the input through the embedding layers
        embedded = self.embedding_layers(x)

        # Pass the embedded input through the recurrent layers (LSTM)
        out, _ = self.rnn(embedded)

        # Apply sigmoid non-linearity to the output of the recurrent layers
        risk_scores = self.sigmoid(out)
        return risk_scores