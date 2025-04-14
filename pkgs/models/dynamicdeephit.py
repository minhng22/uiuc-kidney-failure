import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicDeepHit(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_risks, dropout_lstm=0.2, dropout_cause=0.2):
        super(DynamicDeepHit, self).__init__()
        self.num_risks = num_risks
        self.pred_times = 365 * 15
        
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
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_cause))
                prev_dim = hidden_dim
        
        self.cause_specific_fc = nn.Sequential(*layers) if layers else nn.Identity()
        
        self.risk_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, self.pred_times),
            ) for _ in range(num_risks)
        ])
    
    def attention_net(self, fc_output, mask):
        attention_weights = self.attention(fc_output)
        mask = mask.unsqueeze(-1)
        attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * fc_output, dim=1)
        return context, attention_weights
    
    def forward(self, x, mask, debug_modes=False):
        if debug_modes:
            print(f"x shape: {x.shape}")
            print(f"mask shape: {mask.shape}")

        lstm_output, _ = self.lstm(x)
        if debug_modes:
            print(f"lstm_output shape: {lstm_output.shape}")
        
        fc_output = self.fc(lstm_output)
        if debug_modes:
            print(f"fc_output shape: {fc_output.shape}")

        context, attention_weights = self.attention_net(fc_output, mask)
        if debug_modes:
            print(f"context shape: {context.shape}")
            print(f"attention_weights shape: {attention_weights.shape}")
        
        x = self.cause_specific_fc(context)
        if debug_modes:
            print(f"x shape after cause_specific_fc: {x.shape}")
        
        hazard_preds = [torch.sigmoid(risk_head(x)) for risk_head in self.risk_heads]
        if debug_modes:
            print(f"hazard_preds shape: {[pred.shape for pred in hazard_preds]}")
        
        res = torch.stack(hazard_preds, dim=1)
        if debug_modes:
            print(f"res shape: {res.shape}")

        return res, attention_weights