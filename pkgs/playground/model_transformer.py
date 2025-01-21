import torch
import torch.nn as nn

class TransformerHead(nn.Module):
    """Risk-specific Transformer-based head."""
    def __init__(self, hidden_dim, time_bins):
        super(TransformerHead, self).__init__()
        self.fc = nn.Linear(hidden_dim, time_bins)
        self.norm = nn.BatchNorm1d(time_bins)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.norm(x)
        return torch.sigmoid(x)

class DynamicTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_risks, time_bins, num_heads=4, num_transformer_layers=2, dropout_rate=0.2):
        super(DynamicTransformer, self).__init__()
        self.num_risks = num_risks
        self.time_bins = time_bins

        # Shared Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,  # Input dimensionality
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,  # Feedforward network dimension
            dropout=dropout_rate,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Risk-specific Transformer heads
        self.risk_heads = nn.ModuleList([TransformerHead(input_dim, time_bins) for _ in range(num_risks)])
        
    def forward(self, x, mask):
        # Prepare mask for Transformer (True indicates position to ignore)
        transformer_mask = mask == 0

        # Transpose input for Transformer (required shape: [seq_len, batch_size, input_dim])
        x = x.permute(1, 0, 2)

        # Apply shared Transformer encoder
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=transformer_mask)

        # Aggregate over time (mean pooling)
        aggregated_output = transformer_output.mean(dim=0)  # Shape: [batch_size, input_dim]

        # Compute risk-specific hazard predictions
        hazard_preds = []
        for risk_head in self.risk_heads:
            risk_output = risk_head(aggregated_output)
            hazard_preds.append(risk_output)

        return torch.stack(hazard_preds, dim=1)  # Shape: [batch_size, num_risks, time_bins]
