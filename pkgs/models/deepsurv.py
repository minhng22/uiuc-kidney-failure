import torch
import torch.nn as nn
import torch.optim as optim

class DeepSurv(nn.Module):
    def __init__(self, input_dim, hidden_dims, drop_outs):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dims (list of int): List with sizes for hidden layers.
        """
        super(DeepSurv, self).__init__()
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_outs[i]))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
