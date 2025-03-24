import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicDeepHit(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_risks, dropout_lstm=0.2, dropout_cause=0.2):
        super(DynamicDeepHit, self).__init__()
        self.num_risks = num_risks
        self.time_bins = 30 # discrete time of one month as the original paper
        
        num_layer_lstm = 2
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims[0],
            num_layers=num_layer_lstm,
            batch_first=True,
            dropout=dropout_lstm,
            bidirectional=True
        )
        
        # FC layer after LSTM
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims[0] * num_layer_lstm, hidden_dims[0]),  # Input is output of bidirectional LSTM
            nn.Tanh()
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], 1)
        )
        
        # Create cause-specific fully connected layers
        layers = []
        prev_dim = hidden_dims[0]
        if len(hidden_dims) > 1:
            for hidden_dim in hidden_dims[1:]:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_cause))
                prev_dim = hidden_dim
        
        self.cause_specific_fc = nn.Sequential(*layers) if layers else nn.Identity()
        
        self.risk_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, self.time_bins),
                nn.BatchNorm1d(self.time_bins)
            ) for _ in range(num_risks)
        ])
    
    def attention_net(self, fc_output, mask):
        attention_weights = self.attention(fc_output)
        mask = mask.unsqueeze(-1)
        attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * fc_output, dim=1)
        return context, attention_weights
    
    def forward(self, x, mask):
        lstm_output, _ = self.lstm(x)
        fc_output = self.fc(lstm_output)
        context, attention_weights = self.attention_net(fc_output, mask)
        
        x = self.cause_specific_fc(context)
        
        hazard_preds = [torch.sigmoid(risk_head(x)) for risk_head in self.risk_heads]
        return torch.stack(hazard_preds, dim=1), attention_weights