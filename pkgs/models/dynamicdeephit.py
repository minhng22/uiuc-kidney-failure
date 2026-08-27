import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicDeepHit(nn.Module):
    """Each risk head outputs a per-subject probability-mass-function (PMF)
    over time (softmax across the pred_times axis, classic DeepHit
    formulation), not independent per-day hazards. This bounds total
    predicted event probability across the whole 15-year horizon at <=1 by
    construction.

    This replaces an earlier per-day-independent-sigmoid parametrization
    that had no such bound: nothing stopped every day's hazard from being
    pushed toward 1 simultaneously, and on high-event-rate scenarios
    (four_features/eight_features rep1, ~85% event rate) that degenerate
    "predict certain event for everyone, every day" solution was the
    cheapest fit combine_loss allowed — collapsing the model to predicting
    the same 1.0 risk for every patient regardless of input features
    (c_index=0.5, Brier near its ceiling; see
    generated_data/rep1/ddh_collapse_fix_report.txt for the full
    investigation, and pkgs/scripts/ddh_collapse_fix_experiment.py's git
    history for the 3-way candidate-fix comparison this won). Downstream
    consumers must read this output as a PMF (cumsum over time = CIF, the
    cumulative incidence function) rather than as a hazard curve
    (cumprod(1-hazard)) — see combine_loss_pmf() in
    pkgs/experiments/utils.py and dynamic_deephit_predictions() in
    pkgs/data_analysis/clinical_validity_analysis.py.

    NOTE: every previously-trained *_ddh_model.pt file (any scenario, any
    rep) was trained under the OLD sigmoid-hazard parametrization and is
    NOT compatible with this forward() — loading one of those files and
    calling forward() on it applies today's softmax logic to weights that
    were never trained for it, producing meaningless output. Any such
    stale file must be deleted and retrained (dynamic_deephit.py's run()
    already does this automatically whenever the saved-model path doesn't
    exist)."""

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
        
        # softmax across time (dim=-1), not sigmoid per-day: normalizes each
        # risk head's output into a genuine per-subject PMF over the
        # pred_times axis, so total probability mass is bounded at <=1 by
        # construction (see class docstring for why this matters).
        pmf_preds = [torch.softmax(risk_head(x), dim=-1) for risk_head in self.risk_heads]
        if debug_modes:
            print(f"pmf_preds shape: {[pred.shape for pred in pmf_preds]}")

        res = torch.stack(pmf_preds, dim=1)
        if debug_modes:
            print(f"res shape: {res.shape}")

        return res, attention_weights