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
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :d_model // 2]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class HazardTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_risks, num_layers, nhead, dropout):
        super(HazardTransformer, self).__init__()
        self.num_risks = num_risks
        self.d_model = d_model
        self.max_time = 365 * 15
        
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        self.time_encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        self.pos_encoder = PositionalEncoding(d_model, 1000)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        
        self.hazard_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1)
            ) for _ in range(num_risks)
        ])
    
    def forward(self, features, mask):
        batch_size = features.size(0)
        
        feat_emb = self.input_embedding(features)
        
        mask_expanded = mask.unsqueeze(-1)
        masked_feat_emb = feat_emb * mask_expanded
        
        pooled = torch.sum(masked_feat_emb, dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        
        # use fixed time point as described in the paper
        # "All models learnt from input singleton-length sequences and produced cause-specific hazard predictions as a fixed-length time series."
        eval_times = torch.linspace(0, self.max_time, 1, device=features.device)
        eval_times = eval_times.unsqueeze(0).repeat(batch_size, 1)
        
        num_eval_points = eval_times.size(1)
        
        pooled_expanded = pooled.unsqueeze(1).repeat(1, num_eval_points, 1)
        
        times_expanded = eval_times.unsqueeze(-1)
        time_encoding = self.time_encoder(times_expanded)
        
        combined = pooled_expanded + time_encoding
        
        src = self.pos_encoder(combined)
        
        src = src.transpose(0, 1)
        
        transformer_mask = None
        encoded = self.transformer_encoder(src, mask=transformer_mask)
        
        encoded = encoded.transpose(0, 1)
        
        hazard_outputs = []
        for risk_decoder in self.hazard_decoders:
            hazard = torch.sigmoid(risk_decoder(encoded))
            hazard_outputs.append(hazard.squeeze(-1))
        
        hazard_preds = torch.stack(hazard_outputs, dim=1)
        
        return hazard_preds, encoded, eval_times