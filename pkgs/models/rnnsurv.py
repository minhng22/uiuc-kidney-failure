import torch
import torch.nn as nn

class RNNSurv(nn.Module):
    def __init__(self, input_size, embedding_size, num_embedding_layers, hidden_size, num_recurrent_layers, num_time_intervals):
        """
        Initializes the RNNSurv model.

        Args:
            input_size (int): Number of input features (including time interval identifier).
            embedding_size (int): Dimensionality of the embedding layers.
            num_embedding_layers (int): Number of embedding layers (N1).
            hidden_size (int): Size of the LSTM hidden state.
            num_recurrent_layers (int): Number of LSTM layers (N2).
            num_time_intervals (int): Number of discrete time intervals (K) for survival prediction.
        """
        super(RNNSurv, self).__init__()
        self.embedding_layers = nn.Sequential()
        for i in range(num_embedding_layers):
            input_dim = input_size if i == 0 else embedding_size
            self.embedding_layers.add_module(f'embedding_{i}', nn.Linear(input_dim, embedding_size))
            if i < num_embedding_layers - 1:
                self.embedding_layers.add_module(f'relu_embedding_{i}', nn.ReLU())

        self.rnn = nn.LSTM(embedding_size, hidden_size, num_recurrent_layers, batch_first=True)
        
        # Output layer to predict survival probability for each time interval
        self.output_layer = nn.Linear(hidden_size, num_time_intervals)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass the input through the embedding layers
        embedded = self.embedding_layers(x)

        # The recurrent layers (LSTM)
        out, _ = self.rnn(embedded)

        # Predict survival probabilities for each time interval
        survival_probabilities_logits = self.output_layer(out)
        survival_probabilities = self.sigmoid(survival_probabilities_logits)

        # The paper suggests the risk score is a linear combination of the survival
        # function estimates. where wk for k= 1,...,K are the parameters of the last layer of rnn-surv.
        last_time_step_survival_probs = survival_probabilities[:, -1, :]
        risk_scores = torch.sum(1 - last_time_step_survival_probs, dim=1, keepdim=True)

        return survival_probabilities, risk_scores