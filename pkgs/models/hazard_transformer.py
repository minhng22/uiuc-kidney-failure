import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class HazardTransformer(nn.Module):
    def __init__(self, input_dim, d_model, time_bins, num_risks, num_layers, nhead, dropout):
        super(HazardTransformer, self).__init__()
        self.time_bins = time_bins
        self.num_risks = num_risks
        self.d_model = d_model
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.time_embedding = nn.Embedding(time_bins, d_model)
        self.pos_encoder = PositionalEncoding(d_model, time_bins)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.decoder = nn.Linear(d_model, num_risks)
    
    def forward(self, features, mask):
        batch_size = features.size(0)
        feat_emb = self.input_embedding(features)
        mask = mask.unsqueeze(-1)

        pooled = torch.sum(feat_emb * mask, dim=1) / (mask.sum(dim=1) + 1e-8)
        pooled_expanded = pooled.unsqueeze(1).repeat(1, self.time_bins, 1)
        time_indices = torch.arange(self.time_bins, device=features.device).unsqueeze(0).repeat(batch_size, 1)
        
        t_emb = self.time_embedding(time_indices)
        src = pooled_expanded + t_emb
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)
        encoded = self.transformer_encoder(src)
        encoded = encoded.transpose(0, 1)
        hazard_logits = self.decoder(encoded)
        hazard_probs = torch.sigmoid(hazard_logits)
        hazard_preds = hazard_probs.transpose(1, 2)
        return hazard_preds, encoded
