import torch
import torch.nn as nn
import torch.nn.functional as F


class BranchNet(nn.Module):
    """Branch network for processing input functions (covariate histories)

    NOTE: This preserves your original class name and forward signature.
    Internally we expose `self.network` (as in your original code) so that
    DeepONet can compute per-time-step embeddings by calling `self.network`
    on flattened per-step inputs.
    """
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
        Original pooling behavior is preserved for the batch-aligned case.
        Args:
            u: (batch_size, seq_len, input_dim)
        Returns:
            pooled (batch_size, output_dim)
        """
        batch_size, seq_len, input_dim = u.size()
        u_flat = u.view(-1, input_dim)  # (batch*seq_len, input_dim)
        out = self.network(u_flat)      # (batch*seq_len, feat)
        out = out.view(batch_size, seq_len, -1)
        out = torch.mean(out, dim=1)    # global average pooling
        return out


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
        return self.network(y)


class DeepONet(nn.Module):
    """
    Deep Operator Network adapted to paper semantics while keeping the original
    __init__ and forward signatures.

    - The network's raw forward output is interpreted as log-hazard (h).
    - Use set_time_grid(time_bins) before calling shared-query forward/prediction or compute_survival_loss.
    """

    def __init__(self, input_dim, branch_hidden_dims, trunk_hidden_dims,
                 query_dim=1, dropout=0.1, operator_dim=None):
        # NOTE: signature preserved exactly
        super(DeepONet, self).__init__()

        self.input_dim = input_dim
        self.query_dim = query_dim

        # Branch and trunk (class names preserved)
        self.branch_net = BranchNet(input_dim, branch_hidden_dims, dropout)
        self.trunk_net = TrunkNet(query_dim, trunk_hidden_dims, dropout)

        if operator_dim is None:
            operator_dim = min(self.branch_net.output_dim, self.trunk_net.output_dim)

        self.operator_dim = operator_dim

        self.branch_projection = nn.Linear(self.branch_net.output_dim, operator_dim)
        self.trunk_projection = nn.Linear(self.trunk_net.output_dim, operator_dim)

        self.bias = nn.Parameter(torch.zeros(1))

        # Additional internals required by paper-consistent implementation
        self.time_grid = None              # set with set_time_grid(time_bins) where len(time_bins)==seq_len+1
        self.loghazard_clamp = 20.0        # clamp before exp for numerical stability

    # -------------------------
    # Utility: set time partition (must call if you use shared-query forward or compute_survival_loss)
    # time_bins: 1D tensor of shape (m+1,) with [t0, t1, ..., tm], where seq_len == m
    # -------------------------
    def set_time_grid(self, time_bins: torch.Tensor):
        assert time_bins.ndim == 1, "time_bins must be 1D tensor"
        self.time_grid = time_bins.clone().detach()

    # -------------------------
    # Internal: compute per-query branch encodings by masking future per-step embeddings.
    # Inputs:
    #   u: (B, m, d)
    #   cutoffs: (Q,) int tensor where each cutoff j in [0..m-1] means keep steps 0..j
    # Returns:
    #   branch_proj_out: (B, Q, operator_dim)
    # -------------------------
    def _branch_encodings_per_query(self, u: torch.Tensor, cutoffs: torch.Tensor) -> torch.Tensor:
        B, m, d = u.shape
        device = u.device
        Q = int(cutoffs.numel())

        # compute per-step features using branch_net.network on flattened inputs
        u_flat = u.view(B * m, d)  # (B*m, d)
        per_step = self.branch_net.network(u_flat)  # (B*m, feat)
        feat = per_step.shape[-1]
        per_step = per_step.view(B, m, feat)  # (B, m, feat)

        # build mask (Q, m): keep indices <= cutoff_j
        seq_idx = torch.arange(m, device=device).unsqueeze(0)   # (1, m)
        q_idx = cutoffs.to(device).unsqueeze(1)                # (Q, 1)
        keep_mask = (seq_idx <= q_idx).unsqueeze(0).unsqueeze(-1)  # (1, Q, m, 1)

        # expand features to (B, Q, m, feat), apply mask
        feats_exp = per_step.unsqueeze(1).expand(-1, Q, -1, -1).contiguous()  # (B, Q, m, feat)
        masked = feats_exp * keep_mask  # zero-out future steps

        # sum over time and divide by (j+1) to average across kept steps (avoids dividing by zero)
        denom = (cutoffs.to(masked.dtype) + 1.0).to(device)  # (Q,)
        summed = masked.sum(dim=2)                            # (B, Q, feat)
        averaged = summed / denom.unsqueeze(0).unsqueeze(-1)  # (B, Q, feat)

        # project to operator dim
        BQ_flat = averaged.view(B * Q, feat)
        BQ_proj = self.branch_projection(BQ_flat)             # (B*Q, operator_dim)
        BQ_proj = BQ_proj.view(B, Q, -1)                      # (B, Q, operator_dim)
        return BQ_proj

    # -------------------------
    # forward signature preserved exactly
    # - If y is shared queries (Q, query_dim) and Q != batch_size: produce (batch, Q)
    # - Else (y batch-aligned): produce (batch, 1) as in your original code
    # -------------------------
    def forward(self, u, y):
        batch_size = u.size(0)

        # Shared queries across batch: need to mask branch per-query. Requires time_grid set.
        if y.dim() == 2 and y.size(0) != batch_size:
            if self.time_grid is None:
                raise RuntimeError("time_grid not set. Call set_time_grid(time_bins) before using shared-query forward.")

            # map query times (y) to cutoffs indices on time_grid:
            # time_bins length m+1 where seq_len == m
            time_bins = self.time_grid.to(y.device)
            m_plus1 = time_bins.numel()
            m = m_plus1 - 1
            # bucketize: returns index in [0..m], where bucketize with right=True maps t in (t_j, t_{j+1}] -> idx=j+1
            idx = torch.bucketize(y.squeeze(-1), time_bins, right=True)  # (Q,)
            idx = idx.clamp(min=1, max=m)  # ensure in [1..m]
            cutoffs = (idx - 1).to(torch.long)  # left-index in [0..m-1], shape (Q,)

            # branch encodings per query (B, Q, operator_dim)
            branch_out = self._branch_encodings_per_query(u, cutoffs)  # (B,Q,op_dim)

            # trunk: (Q, query_dim) -> (Q, trunk_out) -> project -> (Q, op_dim)
            trunk_out = self.trunk_net(y)                      # (Q, trunk_out)
            trunk_out = self.trunk_projection(trunk_out)       # (Q, op_dim)

            # expand trunk and compute dot
            trunk_exp = trunk_out.unsqueeze(0).expand(batch_size, -1, -1)  # (B,Q,op_dim)
            operator_output = torch.sum(branch_out * trunk_exp, dim=-1) + self.bias  # (B, Q)
            return operator_output

        else:
            # batch-aligned case: reuse original behavior (global pooling branch)
            branch_output = self.branch_net(u)  # (batch, branch_output_dim)
            branch_output = self.branch_projection(branch_output)  # (batch, operator_dim)

            trunk_output = self.trunk_net(y) if y.dim() == 2 else self.trunk_net(y.unsqueeze(1))
            trunk_output = self.trunk_projection(trunk_output)  # (batch, operator_dim)

            operator_output = torch.sum(branch_output * trunk_output, dim=-1, keepdim=True) + self.bias  # (batch,1)
            return operator_output

    # -------------------------
    # Predict hazard lambda = exp(h) where h is network raw output (log-hazard)
    # keep interface same as Impl A
    # -------------------------
    def predict_hazard(self, u, y):
        # forward returns log-hazard; exponentiate with clamp for numerical stability
        operator_output = self.forward(u, y)
        # clamp then exp
        h_clamped = torch.clamp(operator_output, max=self.loghazard_clamp)
        hazard_rates = torch.exp(h_clamped)
        return hazard_rates

    # -------------------------
    # Predict survival using left-Riemann integration on the time_grid (must be set)
    # This replaces the naive h(t) * t approximation
    # y: (num_queries,1) or (batch,1) as before
    # -------------------------
    def predict_survival(self, u, y):
        if self.time_grid is None:
            raise RuntimeError("time_grid not set. Call set_time_grid(time_bins) before predict_survival.")

        device = u.device
        time_bins = self.time_grid.to(device)
        m_plus1 = time_bins.numel()
        m = m_plus1 - 1
        # left endpoints t0..t_{m-1}
        eval_times = time_bins[:-1].unsqueeze(-1).to(device)  # (m,1)

        # hazards at left endpoints: returns (B, m)
        lam_grid = self.predict_hazard(u, eval_times)  # (B, m)

        # dt per interval (m,)
        dt = (time_bins[1:] - time_bins[:-1]).to(device)

        # cumulative hazard at each left index j: sum_{k=0..j} lam[:,k] * dt[k]
        cumhaz = torch.cumsum(lam_grid * dt.unsqueeze(0), dim=1)  # (B, m)

        # If y shared queries (num_queries,1) map each query to its left index
        if y.dim() == 2 and y.size(0) != u.size(0):
            idx = torch.bucketize(y.squeeze(-1).to(device), time_bins, right=True)  # (Q,)
            idx = idx.clamp(min=1, max=m)
            left_idx = (idx - 1).long()  # (Q,)

            B = u.size(0)
            Q = left_idx.numel()
            S = torch.empty(B, Q, device=device, dtype=lam_grid.dtype)
            # small loop over Q (Q typically manageable); can be vectorized with advanced indexing if needed
            for q, j in enumerate(left_idx.tolist()):
                S[:, q] = torch.exp(-cumhaz[:, j])
            return S

        else:
            # batch-aligned: y is (B,1) or (B,)
            times = y.squeeze(-1).to(device)
            idx = torch.bucketize(times, time_bins, right=True)  # (B,)
            idx = idx.clamp(min=1, max=m)
            left_idx = (idx - 1).long()
            B = u.size(0)
            S = torch.empty(B, device=device, dtype=lam_grid.dtype)
            for i, j in enumerate(left_idx.tolist()):
                S[i] = torch.exp(-cumhaz[i, j])
            return S

    # -------------------------
    # Predict risk score: keep your original method but map to grid if available
    # -------------------------
    def predict_risk_score(self, u):
        reference_time = torch.tensor([[365.0]], device=u.device)
        # use forward behavior unchanged (returns (B,1) if reference_time is expanded appropriately)
        if self.time_grid is not None:
            op = self.forward(u, reference_time)  # will be treated as shared query -> (B,1)
            if op.dim() == 2 and op.size(1) == 1:
                risk_scores = op.squeeze(-1)
            else:
                risk_scores = op
        else:
            # fallback: batch-aligned evaluation
            B = u.size(0)
            ref_batch = reference_time.expand(B, -1).to(u.device)
            out = self.forward(u, ref_batch)  # (B,1)
            risk_scores = out.squeeze(-1)

        # safety checks (preserve similar behavior to Impl A)
        if torch.isnan(risk_scores).any() or torch.isinf(risk_scores).any():
            print("Warning: NaN/Inf detected in risk scores, setting to zeros")
            risk_scores = torch.zeros_like(risk_scores)
        else:
            risk_scores = torch.clamp(risk_scores, -10, 10)

        return risk_scores

    # -------------------------
    # Vectorized discretized negative log-likelihood consistent with the paper (Eq.6)
    # u: (B, m, d) where m == seq_len and time_grid length must be m+1
    # durations: (B,)
    # events: (B,)
    # num_time_points ignored (kept for signature compatibility)
    # -------------------------
    def compute_survival_loss(self, u, durations, events, num_time_points=50):
        if self.time_grid is None:
            raise RuntimeError("time_grid not set. Call set_time_grid(time_bins) before compute_survival_loss.")

        device = u.device
        B, m, d = u.shape
        time_bins = self.time_grid.to(device)
        assert time_bins.numel() == (m + 1), "time_grid length must equal seq_len + 1"

        # Evaluate log-hazard h at left endpoints t0..t_{m-1}: eval_times shape (m,1)
        eval_times = time_bins[:-1].unsqueeze(-1).to(device)
        h_grid = self.forward(u, eval_times)  # (B, m) log-hazard

        # lambda = exp(h) with clamping
        h_clamped = torch.clamp(h_grid, max=self.loghazard_clamp)
        lam_grid = torch.exp(h_clamped)  # (B, m)

        # dt for each left interval j
        dt = (time_bins[1:] - time_bins[:-1]).to(device)  # (m,)

        # indicator_left: I(t_j <= Y_i) where t_j are left endpoints (time_bins[:-1])
        indicator_left = (durations.unsqueeze(1).to(device) >= time_bins[:-1].unsqueeze(0).to(device)).to(dtype=lam_grid.dtype)  # (B, m)

        # integral approx per subject: sum_j indicator_left * lam_grid[:, j] * dt[j]
        integral_terms = (indicator_left * lam_grid * dt.unsqueeze(0)).sum(dim=1)  # (B,)

        # find left index of interval containing duration: idx in [1..m] from bucketize, left_idx = idx-1 in [0..m-1]
        idx = torch.bucketize(durations.to(device), time_bins, right=True)  # (B,)
        idx = idx.clamp(min=1, max=m)
        left_idx = idx - 1  # (B,)

        # h at event left endpoint
        h_at_event = h_grid[torch.arange(B, device=device), left_idx]  # (B,)

        # log-likelihood and mean negative log-likelihood
        loglike = h_at_event * events.to(dtype=h_at_event.dtype) - integral_terms
        nll = -loglike.mean()
        return nll
