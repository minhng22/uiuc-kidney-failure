import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SurvTraceEmbeddings(nn.Module):
    """
    SurvTRACE-style embeddings that handle both categorical and numerical features
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings for categorical features
        if config.get('num_categorical_feature', 0) > 0:
            self.word_embeddings = nn.Embedding(
                config.get('vocab_size', 100), 
                config.get('hidden_size', 128)
            )
        
        # Position embeddings (for sequence positions)
        self.position_embeddings = nn.Embedding(
            config.get('max_position_embeddings', 512),
            config.get('hidden_size', 128)
        )
        
        # Numerical feature projections
        if config.get('num_numerical_feature', 0) > 0:
            self.num_feature_projection = nn.Linear(
                config.get('num_numerical_feature', 1),
                config.get('hidden_size', 128)
            )
        
        # Layer norm and dropout
        self.LayerNorm = nn.LayerNorm(config.get('hidden_size', 128), eps=config.get('layer_norm_eps', 1e-12))
        self.dropout = nn.Dropout(config.get('hidden_dropout_prob', 0.1))
        
    def forward(self, input_ids=None, input_nums=None, position_ids=None, inputs_embeds=None):
        """
        Args:
            input_ids: Categorical feature IDs (batch_size, seq_len, num_cat_features)
            input_nums: Numerical features (batch_size, seq_len, num_num_features) 
            position_ids: Position indices for each sequence position
            inputs_embeds: Pre-computed embeddings
        """
        if inputs_embeds is None:
            embeddings = 0
            
            # Handle categorical features
            if input_ids is not None and hasattr(self, 'word_embeddings'):
                # For multiple categorical features, we need to sum their embeddings
                if input_ids.dim() == 3:  # (batch_size, seq_len, num_cat_features)
                    cat_embeddings = 0
                    for i in range(input_ids.size(-1)):
                        cat_embeddings += self.word_embeddings(input_ids[:, :, i])
                else:  # (batch_size, seq_len)
                    cat_embeddings = self.word_embeddings(input_ids)
                embeddings += cat_embeddings
            
            # Handle numerical features
            if input_nums is not None and hasattr(self, 'num_feature_projection'):
                if input_nums.dim() == 2:  # (batch_size, num_features) -> add seq_len dim
                    input_nums = input_nums.unsqueeze(1)  # (batch_size, 1, num_features)
                num_embeddings = self.num_feature_projection(input_nums)
                embeddings += num_embeddings
        else:
            embeddings = inputs_embeds
            
        # Add position embeddings
        seq_length = embeddings.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=embeddings.device)
            position_ids = position_ids.unsqueeze(0).expand(embeddings.size(0), -1)
            
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class SurvTraceAttention(nn.Module):
    """Multi-head self-attention mechanism based on BERT"""
    
    def __init__(self, config):
        super().__init__()
        if config.get('hidden_size') % config.get('num_attention_heads') != 0:
            raise ValueError(
                f"The hidden size ({config.get('hidden_size')}) is not a multiple of the number of attention "
                f"heads ({config.get('num_attention_heads')})"
            )
        
        self.num_attention_heads = config.get('num_attention_heads')
        self.attention_head_size = int(config.get('hidden_size') / config.get('num_attention_heads'))
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.get('hidden_size'), self.all_head_size)
        self.key = nn.Linear(config.get('hidden_size'), self.all_head_size)
        self.value = nn.Linear(config.get('hidden_size'), self.all_head_size)
        
        self.dropout = nn.Dropout(config.get('attention_probs_dropout_prob', 0.1))
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        return context_layer


class SurvTraceLayer(nn.Module):
    """Single transformer layer based on BERT architecture"""
    
    def __init__(self, config):
        super().__init__()
        self.attention = SurvTraceAttention(config)
        self.intermediate = nn.Linear(config.get('hidden_size'), config.get('intermediate_size'))
        self.output = nn.Linear(config.get('intermediate_size'), config.get('hidden_size'))
        self.LayerNorm_attention = nn.LayerNorm(config.get('hidden_size'), eps=config.get('layer_norm_eps', 1e-12))
        self.LayerNorm_output = nn.LayerNorm(config.get('hidden_size'), eps=config.get('layer_norm_eps', 1e-12))
        self.dropout = nn.Dropout(config.get('hidden_dropout_prob', 0.1))
        
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.LayerNorm_attention(hidden_states + self.dropout(attention_output))
        
        # Feed-forward
        intermediate_output = F.gelu(self.intermediate(attention_output))
        layer_output = self.output(intermediate_output)
        layer_output = self.LayerNorm_output(attention_output + self.dropout(layer_output))
        
        return layer_output


class SurvTraceEncoder(nn.Module):
    """Multi-layer transformer encoder"""
    
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([SurvTraceLayer(config) for _ in range(config.get('num_hidden_layers'))])
        
    def forward(self, hidden_states, attention_mask=None):
        all_hidden_states = []
        
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_hidden_states.append(hidden_states)
            
        return hidden_states, all_hidden_states


class SurvTracePredictionHead(nn.Module):
    """Prediction head for survival analysis"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.get('hidden_size')
        self.num_durations = config.get('num_durations', 5)
        
        # Classification head for discrete time intervals
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()
        self.decoder = nn.Linear(self.hidden_size, self.num_durations)
        
    def forward(self, hidden_states):
        # Use [CLS] token (first token) for classification
        cls_hidden = hidden_states[:, 0]  # (batch_size, hidden_size)
        
        # Apply prediction head
        hidden = self.dense(cls_hidden)
        hidden = self.activation(hidden)
        logits = self.decoder(hidden)  # (batch_size, num_durations)
        
        return logits


class SurvTimeSurvival(nn.Module):
    """
    SurvTimeSurvival model implementation based on SurvTRACE architecture
    for survival analysis with time-varying covariates and multiple visits.
    
    Paper: "SurvTimeSurvival: Survival Analysis On The Patient With Multiple Visits/Records"
    Built upon SurvTRACE: "SurvTRACE: Transformers for Survival Analysis with Competing Events"
    """
    
    def __init__(self, input_dim, num_categorical_feature=0, num_numerical_feature=None, 
                 vocab_size=100, hidden_size=128, num_hidden_layers=3, 
                 num_attention_heads=8, intermediate_size=512, 
                 hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                 num_durations=5, max_position_embeddings=512):
        super(SurvTimeSurvival, self).__init__()
        
        # Set up configuration
        if num_numerical_feature is None:
            num_numerical_feature = input_dim - num_categorical_feature
            
        self.config = {
            'num_categorical_feature': num_categorical_feature,
            'num_numerical_feature': num_numerical_feature,
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_hidden_layers': num_hidden_layers,
            'num_attention_heads': num_attention_heads,
            'intermediate_size': intermediate_size,
            'hidden_dropout_prob': hidden_dropout_prob,
            'attention_probs_dropout_prob': attention_probs_dropout_prob,
            'num_durations': num_durations,
            'max_position_embeddings': max_position_embeddings,
            'layer_norm_eps': 1e-12,
        }
        
        # Model components
        self.embeddings = SurvTraceEmbeddings(self.config)
        self.encoder = SurvTraceEncoder(self.config)
        self.prediction_head = SurvTracePredictionHead(self.config)
        
        # Store duration index for survival predictions
        self.duration_index = None
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def create_attention_mask(self, input_ids, seq_lengths=None):
        """Create attention mask for padding tokens"""
        if seq_lengths is not None:
            batch_size, max_len = input_ids.size(0), input_ids.size(1)
            mask = torch.zeros(batch_size, max_len, device=input_ids.device, dtype=torch.bool)
            
            for i, length in enumerate(seq_lengths):
                if length < max_len:
                    mask[i, length:] = True
                    
            # Convert to attention mask format (add dims for multi-head attention)
            attention_mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, seq_len)
            attention_mask = attention_mask.float() * -10000.0
            return attention_mask
        else:
            return None
        
    def forward(self, input_ids=None, input_nums=None, attention_mask=None, seq_lengths=None):
        """
        Forward pass
        
        Args:
            input_ids: Categorical feature IDs (batch_size, seq_len, num_cat_features) or (batch_size, seq_len)
            input_nums: Numerical features (batch_size, seq_len, num_num_features)
            attention_mask: Attention mask for padding
            seq_lengths: Actual sequence lengths for each sample
            
        Returns:
            hazard_logits: Raw logits for discrete time intervals (batch_size, num_durations)
        """
        # Create attention mask if seq_lengths provided
        if attention_mask is None and seq_lengths is not None:
            # For simplicity, create mask based on input_nums or input_ids
            if input_nums is not None:
                attention_mask = self.create_attention_mask(input_nums[:, :, 0:1], seq_lengths)
            elif input_ids is not None:
                if input_ids.dim() == 3:
                    attention_mask = self.create_attention_mask(input_ids[:, :, 0:1], seq_lengths)
                else:
                    attention_mask = self.create_attention_mask(input_ids, seq_lengths)
        
        # Embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            input_nums=input_nums
        )
        
        # Encoder
        sequence_output, all_hidden_states = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask
        )
        
        # Prediction head
        hazard_logits = self.prediction_head(sequence_output)
        
        return hazard_logits
    
    def predict_hazard(self, input_ids=None, input_nums=None, seq_lengths=None, batch_size=None):
        """Predict hazard rates"""
        if batch_size is None:
            logits = self.forward(input_ids, input_nums, seq_lengths=seq_lengths)
            hazard = F.softplus(logits)
        else:
            # Handle batched prediction
            hazards = []
            num_samples = input_nums.size(0) if input_nums is not None else input_ids.size(0)
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            self.eval()
            with torch.no_grad():
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_samples)
                    
                    batch_input_ids = input_ids[start_idx:end_idx] if input_ids is not None else None
                    batch_input_nums = input_nums[start_idx:end_idx] if input_nums is not None else None
                    batch_seq_lengths = seq_lengths[start_idx:end_idx] if seq_lengths is not None else None
                    
                    batch_logits = self.forward(batch_input_ids, batch_input_nums, seq_lengths=batch_seq_lengths)
                    batch_hazard = F.softplus(batch_logits)
                    hazards.append(batch_hazard)
            
            hazard = torch.cat(hazards, dim=0)
        
        # Add padding at the start for cumulative calculations
        hazard_padded = F.pad(hazard, (1, 0), value=0)  # Add 0 at the beginning
        return hazard_padded
    
    def predict_survival(self, input_ids=None, input_nums=None, seq_lengths=None, batch_size=None):
        """Predict survival probabilities"""
        hazard = self.predict_hazard(input_ids, input_nums, seq_lengths, batch_size)
        
        # Calculate survival probabilities: S(t) = exp(-cumulative_hazard(t))
        cumulative_hazard = hazard.cumsum(dim=1)
        survival = torch.exp(-cumulative_hazard)
        
        return survival
    
    def predict_risk_score(self, input_ids=None, input_nums=None, seq_lengths=None, batch_size=None):
        """Predict risk scores (1 - survival at median time)"""
        survival = self.predict_survival(input_ids, input_nums, seq_lengths, batch_size)
        
        # Use middle time point as reference
        mid_time_idx = survival.size(1) // 2
        risk_scores = 1 - survival[:, mid_time_idx]
        
        return risk_scores
    
    def predict_survival_at_times(self, input_ids=None, input_nums=None, time_points=None, seq_lengths=None):
        """
        Predict survival probabilities at specific time points
        
        Args:
            input_ids: Categorical features
            input_nums: Numerical features  
            time_points: Specific time points to evaluate (tensor or list)
            seq_lengths: Sequence lengths
            
        Returns:
            Survival probabilities at specified time points
        """
        if time_points is None:
            return self.predict_survival(input_ids, input_nums, seq_lengths)
            
        hazard = self.predict_hazard(input_ids, input_nums, seq_lengths)
        
        # Convert time points to indices (assuming time_points are normalized to [0, num_durations])
        if isinstance(time_points, (list, np.ndarray)):
            time_points = torch.tensor(time_points, device=hazard.device)
            
        # Interpolate hazard at specific time points
        num_durations = hazard.size(1) - 1  # Subtract 1 for padding
        time_indices = (time_points * num_durations).long().clamp(0, num_durations-1)
        
        survival_at_times = []
        for t_idx in time_indices:
            cumulative_hazard = hazard[:, :t_idx+1].sum(dim=1)
            survival_at_t = torch.exp(-cumulative_hazard)
            survival_at_times.append(survival_at_t)
            
        return torch.stack(survival_at_times, dim=1)  # (batch_size, num_time_points)
