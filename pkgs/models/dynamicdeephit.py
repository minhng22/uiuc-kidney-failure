import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicDeepHit(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_risks, time_bins, dropout_rate=0.2):
        super(DynamicDeepHit, self).__init__()
        self.num_risks = num_risks
        self.time_bins = time_bins
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims[0],
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dims[0] * 2, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], 1)
        )
        
        self.fc_layers = nn.ModuleList()
        prev_dim = hidden_dims[0] * 2  # * 2 for bidirectional LSTM
        
        for hidden_dim in hidden_dims[1:]:
            self.fc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.risk_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, time_bins),
                nn.BatchNorm1d(time_bins)
            ) for _ in range(num_risks)
        ])
        
    def attention_net(self, lstm_output, mask):
        attention_weights = self.attention(lstm_output)
        mask = mask.unsqueeze(-1)
        attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights
        
    def forward(self, x, mask):
        lstm_output, _ = self.lstm(x)
        
        context, attention_weights = self.attention_net(lstm_output, mask)
        
        x = context
        for layer in self.fc_layers:
            x = layer(x)
        
        hazard_preds = []
        for risk_head in self.risk_heads:
            risk_output = risk_head(x)
            hazard_preds.append(torch.sigmoid(risk_output))
        
        return torch.stack(hazard_preds, dim=1), attention_weights
