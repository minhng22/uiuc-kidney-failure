import torch
import torch.nn as nn
import torch.nn.functional as F

class EventDrivenAttention(nn.Module):
    def __init__(self, d_model, n_heads, event_weighting='gaussian', event_strength=1.0):
        super(EventDrivenAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.event_weighting = event_weighting  
        self.event_strength = event_strength    
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        
        # Learnable positional embeddings based on proximity to event
        self.proximity_embedding = nn.Parameter(torch.randn(d_model))
    
    def event_proximity_bias(self, time_indices, event_indices):
        # Calculate proximity weights based on distance to event
        proximity = torch.abs(time_indices.unsqueeze(-1) - event_indices.unsqueeze(0))  # Time distance to event
        
        if self.event_weighting == 'gaussian':
            # Gaussian decay function
            proximity_weight = torch.exp(-self.event_strength * (proximity.float() ** 2))
        elif self.event_weighting == 'linear':
            # Linear decay function
            proximity_weight = 1 / (1 + self.event_strength * proximity.float())
        else:
            raise ValueError("Unsupported event weighting type")
        
        return proximity_weight
    
    def forward(self, x, event_time_indices):
        """
        x: Input sequence (batch_size, seq_len, d_model)
        event_time_indices: Event occurrence indices per sample in batch (batch_size, 1)
        """
        batch_size, seq_len, _ = x.size()
        
        # Create proximity bias (batch_size, seq_len, seq_len)
        time_indices = torch.arange(seq_len, device=x.device).repeat(batch_size, 1)
        proximity_weights = self.event_proximity_bias(time_indices, event_time_indices)
        
        # Apply event-driven bias to attention mechanism
        attn_output, attn_weights = self.attention(x, x, x)  # Self-attention
        
        # Apply proximity weights to attention scores
        biased_attn_weights = attn_weights * proximity_weights.unsqueeze(1)
        biased_attn_weights = F.softmax(biased_attn_weights, dim=-1)
        
        # Weighted sum of value vectors
        output = torch.matmul(biased_attn_weights, x)
        return output, biased_attn_weights


# Full Model with Transformer Encoder + Event-Driven Attention

class EventDrivenTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers, event_weighting='gaussian', event_strength=1.0):
        super(EventDrivenTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.event_attentions = nn.ModuleList([
            EventDrivenAttention(d_model, n_heads, event_weighting, event_strength) for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, 1)  # For final survival time prediction
    
    def forward(self, x, event_time_indices):
        """
        x: Input sequence (batch_size, seq_len, input_dim)
        event_time_indices: Event occurrence indices per sample in batch (batch_size, 1)
        """
        x = self.embedding(x)  # Embed input features
        
        for attention_layer in self.event_attentions:
            x, _ = attention_layer(x, event_time_indices)  # Apply event-driven attention at each layer
        
        # Global pooling and prediction
        out = x.mean(dim=1)  # Global average pooling across time steps
        out = self.fc_out(out)  # Final prediction (survival time / risk)
        return out