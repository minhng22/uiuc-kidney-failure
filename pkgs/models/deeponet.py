import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BranchNet(nn.Module):
    """Branch network for processing input functions (covariate histories)"""
    
    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        super(BranchNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim
        
    def forward(self, u):
        """
        Args:
            u: Input functions/covariate histories (batch_size, seq_len, input_dim)
        Returns:
            Branch network output (batch_size, output_dim)
        """
        # Process each time step and aggregate
        batch_size, seq_len, input_dim = u.size()
        u_flat = u.view(-1, input_dim)  # (batch_size * seq_len, input_dim)
        
        # Pass through network
        output = self.network(u_flat)  # (batch_size * seq_len, output_dim)
        output = output.view(batch_size, seq_len, -1)  # (batch_size, seq_len, output_dim)
        
        # Global average pooling over time dimension
        output = torch.mean(output, dim=1)  # (batch_size, output_dim)
        
        return output


class TrunkNet(nn.Module):
    """Trunk network for processing query points (time points)"""
    
    def __init__(self, query_dim, hidden_dims, dropout=0.1):
        super(TrunkNet, self).__init__()
        
        layers = []
        prev_dim = query_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim
        
    def forward(self, y):
        """
        Args:
            y: Query points/time points (batch_size, query_dim) or (num_queries, query_dim)
        Returns:
            Trunk network output (batch_size, output_dim) or (num_queries, output_dim)
        """
        return self.network(y)


class DeepONet(nn.Module):
    """
    Deep Operator Network (DeepONet) for survival analysis with time-varying covariates.
    
    Based on the paper: "Nonparametric Estimation of Conditional Survival Function 
    with Time-Varying Covariates Using DeepONet"
    
    The model learns a nonlinear operator that maps covariate histories to 
    conditional hazard/survival functions.
    """
    
    def __init__(self, input_dim, branch_hidden_dims, trunk_hidden_dims, 
                 query_dim=1, dropout=0.1, operator_dim=None):
        super(DeepONet, self).__init__()
        
        self.input_dim = input_dim
        self.query_dim = query_dim
        
        # Branch network processes covariate histories
        self.branch_net = BranchNet(input_dim, branch_hidden_dims, dropout)
        
        # Trunk network processes query time points
        self.trunk_net = TrunkNet(query_dim, trunk_hidden_dims, dropout)
        
        # Ensure branch and trunk networks have compatible output dimensions
        if operator_dim is None:
            operator_dim = min(self.branch_net.output_dim, self.trunk_net.output_dim)
        
        self.operator_dim = operator_dim
        
        # Final projection layers to ensure compatible dimensions
        self.branch_projection = nn.Linear(self.branch_net.output_dim, operator_dim)
        self.trunk_projection = nn.Linear(self.trunk_net.output_dim, operator_dim)
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, u, y):
        """
        Forward pass of DeepONet - standard operator learning formulation
        
        Args:
            u: Input functions/covariate histories (batch_size, seq_len, input_dim)
            y: Query points/time points (batch_size, query_dim) or (num_queries, query_dim)
            
        Returns:
            operator_output: Operator output G(u)(y) representing hazard at query times
        """
        # Branch network output
        branch_output = self.branch_net(u)  # (batch_size, branch_output_dim)
        branch_output = self.branch_projection(branch_output)  # (batch_size, operator_dim)
        
        # Trunk network output  
        trunk_output = self.trunk_net(y)  # (batch_size or num_queries, trunk_output_dim)
        trunk_output = self.trunk_projection(trunk_output)  # (batch_size or num_queries, operator_dim)
        
        # Compute operator output: dot product + bias (standard DeepONet formulation)
        if y.dim() == 2 and y.size(0) != u.size(0):
            # y is (num_queries, query_dim), need to broadcast
            num_queries = y.size(0)
            batch_size = u.size(0)
            
            # Expand branch output: (batch_size, operator_dim) -> (batch_size, num_queries, operator_dim)
            branch_expanded = branch_output.unsqueeze(1).expand(-1, num_queries, -1)
            
            # Expand trunk output: (num_queries, operator_dim) -> (batch_size, num_queries, operator_dim)  
            trunk_expanded = trunk_output.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Element-wise multiplication and sum over operator dimension + bias
            operator_output = torch.sum(branch_expanded * trunk_expanded, dim=-1) + self.bias  # (batch_size, num_queries)
            
        else:
            # y has same batch size as u, direct dot product + bias
            operator_output = torch.sum(branch_output * trunk_output, dim=-1, keepdim=True) + self.bias  # (batch_size, 1)
            
        return operator_output
    
    def predict_hazard(self, u, y):
        """
        Predict conditional hazard function h(t|u) at query time points
        
        Args:
            u: Covariate histories (batch_size, seq_len, input_dim)
            y: Query time points (num_queries, 1)
            
        Returns:
            Conditional hazard rates (batch_size, num_queries)
        """
        # Get operator output
        operator_output = self.forward(u, y)
        
        # Apply softplus to ensure positive hazard rates
        hazard_rates = F.softplus(operator_output)
        
        return hazard_rates
    
    def predict_survival(self, u, y):
        """
        Predict conditional survival function S(t|u) at query time points
        
        Args:
            u: Covariate histories (batch_size, seq_len, input_dim)  
            y: Query time points (num_queries, 1)
            
        Returns:
            Conditional survival probabilities (batch_size, num_queries)
        """
        # Get hazard rates
        hazard_rates = self.predict_hazard(u, y)
        
        # Convert to survival probabilities using: S(t) = exp(-∫h(s)ds)
        # For discrete case: S(t) ≈ exp(-h(t) * t)
        if y.dim() == 2:
            time_points = y.squeeze(-1)  # (num_queries,)
            time_points = time_points.unsqueeze(0)  # (1, num_queries)
        else:
            time_points = y.unsqueeze(0)
            
        # Calculate cumulative hazard (approximated)
        cumulative_hazard = hazard_rates * time_points
        
        # Survival probability
        survival_probs = torch.exp(-cumulative_hazard)
        
        return survival_probs
    
    def predict_risk_score(self, u):
        """
        Predict risk scores for patients based on their covariate histories
        
        Args:
            u: Covariate histories (batch_size, seq_len, input_dim)
            
        Returns:
            Risk scores (batch_size,)
        """
        # Use a reference time point (e.g., median follow-up time)
        reference_time = torch.tensor([[365.0]], device=u.device)  # 1 year
        
        # Get operator output at reference time
        risk_scores = self.forward(u, reference_time).squeeze(-1)
        
        # Check for NaN/inf values and clamp to reasonable range
        if torch.isnan(risk_scores).any() or torch.isinf(risk_scores).any():
            print("Warning: NaN/Inf detected in risk scores, setting to zeros")
            risk_scores = torch.zeros_like(risk_scores)
        else:
            # Clamp to prevent extreme values
            risk_scores = torch.clamp(risk_scores, -10, 10)
        
        return risk_scores
    
    def compute_survival_loss(self, u, durations, events, num_time_points=50):
        """
        Compute survival loss based on likelihood for censored data
        
        Args:
            u: Covariate histories (batch_size, seq_len, input_dim)
            durations: Event/censoring times (batch_size,)
            events: Event indicators (batch_size,)
            num_time_points: Number of time points for evaluation
            
        Returns:
            Loss value
        """
        batch_size = u.size(0)
        device = u.device
        
        # Create time points for evaluation
        max_time = durations.max().item()
        time_points = torch.linspace(1, max_time, num_time_points, device=device).unsqueeze(-1)
        
        # Get hazard rates at all time points
        hazard_rates = self.predict_hazard(u, time_points)  # (batch_size, num_time_points)
        
        # Approximate survival probabilities
        dt = max_time / num_time_points
        cumulative_hazard = torch.cumsum(hazard_rates * dt, dim=1)
        survival_probs = torch.exp(-cumulative_hazard)
        
        # Compute likelihood for each patient
        total_loss = 0.0
        
        for i in range(batch_size):
            t_i = durations[i].item()
            delta_i = events[i].item()
            
            # Find closest time index
            time_idx = min(int(t_i / dt), num_time_points - 1)
            
            if delta_i == 1:  # Event occurred
                # Likelihood: h(t_i) * S(t_i)
                h_t = hazard_rates[i, time_idx]
                S_t = survival_probs[i, time_idx]
                likelihood = h_t * S_t
            else:  # Censored
                # Likelihood: S(t_i)
                likelihood = survival_probs[i, time_idx]
            
            # Add negative log-likelihood
            total_loss -= torch.log(likelihood + 1e-8)
        
        return total_loss / batch_size
