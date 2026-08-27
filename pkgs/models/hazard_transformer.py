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
    """Each risk decoder outputs a per-subject probability-mass-function
    (PMF) over its num_time_bins time bins (softmax across the bin axis),
    not independent per-bin hazards. This bounds total predicted event
    probability across all bins at <=1 by construction -- the same fix
    applied to DynamicDeepHit (see pkgs/models/dynamicdeephit.py's class
    docstring and generated_data/rep1/ddh_collapse_fix_report.txt).

    This replaces an earlier per-bin-independent-sigmoid parametrization
    that had no such bound. Confirmed to still collapse even after fixing a
    separate masking bug in the training loss (see
    pkgs/experiments/hazard_transformer.py's objective(), "INCLUSIVE" mask
    comment): a fresh rep99 retrain under the mask fix alone still collapsed
    2 of 3 scenarios (eight_features, twenty_features_heterogeneous both
    landed on c_index=0.5, predicted risk constant across every patient) --
    proof the mask fix wasn't sufficient and the architecture itself needed
    the same bounded-PMF fix DDH got.

    Downstream consumers must read this output as a PMF (cumsum over bins =
    CIF) rather than as a hazard curve (cumprod(1-hazard)) -- see
    hazard_transformer_pmf_loss() and c_idx()/auc()/brier_score_evaluation()
    in pkgs/experiments/hazard_transformer.py, and
    hazard_transformer_predictions() in
    pkgs/data_analysis/clinical_validity_analysis.py.

    NOTE: every previously-trained *_hazard_transformer_model.pt file (any
    scenario, any rep) was trained under the OLD sigmoid parametrization and
    is NOT compatible with this forward() -- must be deleted and retrained."""

    def __init__(self, input_dim, d_model, num_risks, num_layers, nhead, dropout, num_time_bins=100):
        super(HazardTransformer, self).__init__()
        self.num_risks = num_risks
        self.d_model = d_model
        self.max_time = 730
        # discretize the follow-up horizon into bins; a single bin collapses the model to one time point
        self.num_time_bins = num_time_bins
        
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
        
        # discrete-time hazard prediction across a fixed-length series of time bins, as described in the paper
        # "All models learnt from input singleton-length sequences and produced cause-specific hazard predictions as a fixed-length time series."
        eval_times = torch.linspace(0, self.max_time, self.num_time_bins, device=features.device)
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
        
        # softmax across the time-bin axis (dim=-2, since encoded is
        # (batch, num_time_bins, d_model) transposed back to
        # (batch, num_time_bins, 1) by risk_decoder -- softmax over bins,
        # not over the singleton last dim), not sigmoid per-bin: normalizes
        # each risk decoder's output into a genuine per-subject PMF over
        # time (see class docstring).
        pmf_outputs = []
        for risk_decoder in self.hazard_decoders:
            logits = risk_decoder(encoded).squeeze(-1)  # (batch, num_time_bins)
            pmf_outputs.append(torch.softmax(logits, dim=-1))

        pmf_preds = torch.stack(pmf_outputs, dim=1)

        return pmf_preds, encoded, eval_times