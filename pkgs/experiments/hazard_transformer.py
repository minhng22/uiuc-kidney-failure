import math
import pandas as pd
from pkgs.commons import egfr_tv_hazard_transformer_model_path,  hg_hazard_transformer_model_path, egfr_components_hazard_transformer_model_path, fivelabms_hazard_transformer_model_path, ckd_fifty_features_heterogeneous_hazard_transformer_model_path, four_features_hazard_transformer_model_path, eight_features_hazard_transformer_model_path, twenty_features_heterogeneous_hazard_transformer_model_path, ckd_fifty_features_heterogeneous_train_data_path
from pkgs.data_analysis.model_data_store import get_train_test_data
from pkgs.models.hazard_transformer import HazardTransformer, HazardTransformerDataset, custom_collate_fn
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from pkgs.experiments.utils import ex_optuna, get_tv_rnn_model_features, combine_loss, compute_brier_score_from_risk_scores
from pkgs.data_analysis.types import ExperimentScenario
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from lifelines.utils import concordance_index
from pkgs.experiments.utils import get_device

num_risks = 1
NUM_TIME_BINS = 100
# Fixed common horizon (days) for scalar c-index/AUC/Brier evaluation below --
# NOT each patient's own observed time. Reading CIF at each patient's own time
# was tried first (to dodge the "final-bin CIF is always 1.0 for every
# softmax-normalized PMF" degeneracy) but introduced a worse confound: CIF only
# grows over time, so a patient followed LONGER (typically the healthier,
# lower-risk ones, especially censored) automatically gets evaluated at a
# LATER bin where cumulative risk has grown more, regardless of true risk --
# this systematically inflates long-survivors' scores and deflates
# short-survivors', independent of how good the model actually is. Confirmed
# empirically on the trained eight_features/rep99 model: own-time c-index was
# 0.148 (badly inverted), a FIXED common horizon gave 0.477 (still modest, but
# a real, unconfounded reading) -- same direction of fix DDH's
# dynamic_deephit_predictions() already uses (day 730, fixed for everyone).
# 365 (not 730, this model's own max_time) specifically avoids the *separate*
# final-bin-saturation issue: bins span exactly [0, max_time], so day 730
# lands on the very last bin, where cumsum(softmax(...)) == 1.0 for everyone
# by construction -- also degenerate, just for a different reason.
EVAL_HORIZON_DAYS = 365


def _fixed_horizon_bin_idx(num_bins, max_time, device):
    """Same bin index for every patient in the batch -- see EVAL_HORIZON_DAYS."""
    return int(round(min(EVAL_HORIZON_DAYS, max_time) / max_time * (num_bins - 1)))


def hazard_loss(hazard_preds, delta, time_mask, eps=1e-7):
    p = hazard_preds.clamp(min=eps, max=1-eps)           
    ll1 = delta * torch.log(p)                           
    ll0 = (1 - delta) * torch.log(1 - p)                 
    neg_ll = - (ll1 + ll0) * time_mask.unsqueeze(1)      
    return neg_ll.sum() / (time_mask.sum() * hazard_preds.size(1) + eps)


def hazard_transformer_pmf_loss(pmf_preds, time_intervals, event_indicators, max_time, eps=1e-7):
    """Replaces hazard_loss() now that the model outputs a softmax-normalized
    PMF over time bins instead of independent per-bin sigmoid hazards (see
    HazardTransformer's class docstring in pkgs/models/hazard_transformer.py).
    Mirrors combine_loss_pmf() in pkgs/experiments/utils.py (used for
    DynamicDeepHit), adapted to this model's day->bin_idx mapping (100 bins
    spanning [0, max_time] days, not one bin per literal day). Per-day PMF
    mass already IS P(T=bin) directly (softmax normalizes across all bins at
    once), so unlike the old hazard_loss() there's no separate mask/delta
    construction needed -- P(event by bin) = cumsum(pmf), matching combine_loss_pmf's
    event_log_prob/censor_log_prob structure exactly."""
    batch, _, T = pmf_preds.shape
    risk_pmf = pmf_preds[:, 0, :]  # single risk (ESRD)
    event_indicators = event_indicators.view(-1).float()

    bin_idx = torch.clamp((time_intervals.view(-1).float() / max_time * (T - 1)).round().long(), min=0, max=T - 1)
    idx = torch.arange(batch, device=pmf_preds.device)

    cif = torch.cumsum(risk_pmf, dim=1)

    pmf_at_t = risk_pmf[idx, bin_idx].clamp(min=eps)
    event_log_prob = torch.log(pmf_at_t) * event_indicators

    surv_at_t = (1.0 - cif[idx, bin_idx]).clamp(min=eps)
    censor_log_prob = torch.log(surv_at_t) * (1 - event_indicators)

    return -torch.mean(event_log_prob + censor_log_prob)

def objective(trial, scenario_name: ExperimentScenario):
    device = get_device()

    print(f"Running trial {trial.number} for {scenario_name} on device {device}")
    df, _ = get_train_test_data(scenario_name)

    dataset = HazardTransformerDataset(df, scenario_name)
    train_loader = DataLoader(dataset, shuffle=True, collate_fn=custom_collate_fn, batch_size=256)

    input_dim = len(get_tv_rnn_model_features(scenario_name))
    num_layers = trial.suggest_int("num_layers", 2, 6)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    drop_out = trial.suggest_float('drop_out_rate', 0.1, 0.5)
    num_epochs = 50
    nhead = trial.suggest_int("n_head", 1, 8)
    nhead_factor = trial.suggest_int("nhead_factor", 1, 16)
    hidden_dims = nhead * nhead_factor

    model = HazardTransformer(input_dim, hidden_dims, num_risks, num_layers, nhead, drop_out, num_time_bins=NUM_TIME_BINS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping parameters
    patience = 5
    best_loss = float('inf')
    patience_counter = 0

    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        total_loss = 0
        for i, (features, mask, time_intervals, event_indicators, _, _) in enumerate(train_loader):
            features, mask, time_intervals, event_indicators = [x.to(device) for x in (features, mask, time_intervals, event_indicators)]
            optimizer.zero_grad()

            pmf_preds, _, _ = model(features, mask)

            # PMF-based loss (see HazardTransformer's class docstring and
            # hazard_transformer_pmf_loss()'s docstring) -- replaces the old
            # per-bin hazard_loss() + delta/time_mask construction entirely;
            # a PMF's per-bin mass already IS P(T=bin) directly, no separate
            # masking needed.
            loss = hazard_transformer_pmf_loss(pmf_preds, time_intervals, event_indicators, model.max_time)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Check for early stopping
        avg_loss = total_loss / len(train_loader)
        print(f'Average Loss: {avg_loss}')
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'Patience Counter: {patience_counter}')

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    c_index = c_idx(model, DataLoader(dataset, shuffle=True, collate_fn=custom_collate_fn, batch_size=256), df, device)
    trial.set_user_attr(key="model", value=model)
    return c_index

def c_idx(model, data_loader, train_df, device):
    model.eval()
    y_train = Surv.from_arrays(
        event=train_df['has_esrd'].values.astype(bool),
        time=train_df['duration_in_days'].values,
        name_event='has_esrd',
        name_time='duration_in_days')

    all_scores, all_times, all_events = [], [], []

    with torch.no_grad():
        for X, mask, times, events, _, _ in data_loader:
            X, mask = X.to(device), mask.to(device)
            pmf_preds, _, _ = model(X, mask)
            # Fixed common horizon for every patient -- see EVAL_HORIZON_DAYS
            # above for why (own-observed-time was confounded by follow-up
            # duration itself, not just the final-bin degeneracy it was meant
            # to dodge).
            cif = torch.cumsum(pmf_preds[:, 0, :], dim=1)
            bin_idx = _fixed_horizon_bin_idx(cif.size(1), model.max_time, device)
            surv_at_horizon = 1 - cif[:, bin_idx]

            all_scores.extend(surv_at_horizon.cpu().tolist())
            all_times.extend(times.squeeze(1).cpu().numpy().tolist())
            all_events.extend(events.squeeze(1).cpu().numpy().tolist())

    c_td = concordance_index(all_times, all_scores, all_events)
    print(f"C-index: {c_td:.4f}")
    return c_td

def auc(model: HazardTransformer, train_df, dataloader: DataLoader, device):
    y_train = Surv.from_arrays(
        event=train_df['has_esrd'].values, time=train_df['duration_in_days'].values, name_event='has_esrd', name_time='duration_in_days')
    aucs = []
    times = np.arange(1, 730, 1)
    for features, mask, time_to_events, event_indicators, _, _ in dataloader:
        features, mask = features.to(device), mask.to(device)
        y_test = Surv.from_arrays(
            event=event_indicators.squeeze(),
            time=time_to_events.squeeze(),
            name_event='has_esrd',
            name_time='duration_in_days'
        )

        pmf_preds, _, _ = model(features, mask)
        # Fixed common horizon -- see EVAL_HORIZON_DAYS.
        cif = torch.cumsum(pmf_preds[:, 0, :], dim=1)
        bin_idx = _fixed_horizon_bin_idx(cif.size(1), model.max_time, device)
        risk_scores = cif[:, bin_idx].detach().cpu().numpy()

        _, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)
        aucs.append(mean_auc)

    avg_auc = np.mean(aucs, axis=0)
    print(f"Mean time-dependent AUC: {avg_auc:.2f}")

def brier_score_evaluation(model: HazardTransformer, train_df, dataloader: DataLoader, device):
    """Compute Brier Score for Hazard Transformer model"""
    all_risk_scores = []
    all_times = []
    all_events = []
    
    for features, mask, time_to_events, event_indicators, _, _ in dataloader:
        features, mask = features.to(device), mask.to(device)
        
        pmf_preds, _, _ = model(features, mask)
        # Fixed common horizon -- see EVAL_HORIZON_DAYS.
        cif = torch.cumsum(pmf_preds[:, 0, :], dim=1)
        bin_idx = _fixed_horizon_bin_idx(cif.size(1), model.max_time, device)
        risk_scores = cif[:, bin_idx].detach().cpu().numpy()

        all_risk_scores.extend(risk_scores)
        all_times.extend(time_to_events.squeeze().numpy())
        all_events.extend(event_indicators.squeeze().numpy())
    
    # Create test dataframe from collected data
    test_df = pd.DataFrame({
        'duration_in_days': all_times,
        'has_esrd': all_events
    })
    
    # Compute Brier Score
    brier_score = compute_brier_score_from_risk_scores(train_df, test_df, np.array(all_risk_scores))
    if brier_score is not None:
        print(f'Integrated Brier Score Test: {brier_score}')
    
    return brier_score


def run(scenario_name: ExperimentScenario):
    device = get_device()
    df, df_test = get_train_test_data(scenario_name)

    model_saved_path_dict = {
        ExperimentScenario.TIME_VARIANT: egfr_tv_hazard_transformer_model_path,
        ExperimentScenario.HETEROGENEOUS: hg_hazard_transformer_model_path,
        ExperimentScenario.EGFR_COMPONENTS: egfr_components_hazard_transformer_model_path,
        ExperimentScenario.FIVELABMS: fivelabms_hazard_transformer_model_path,
        ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS: ckd_fifty_features_heterogeneous_hazard_transformer_model_path,
        ExperimentScenario.FOUR_FEATURES: four_features_hazard_transformer_model_path,
        ExperimentScenario.EIGHT_FEATURES: eight_features_hazard_transformer_model_path,
        ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS: twenty_features_heterogeneous_hazard_transformer_model_path,
    }
    model_saved_path = model_saved_path_dict[scenario_name]
    
    if os.path.exists(model_saved_path) and getattr(torch.load(model_saved_path, map_location='cpu', weights_only=False), 'architecture_version', 1) == HazardTransformer.ARCHITECTURE_VERSION:
        print("Loading from saved weights")
        model = torch.load(model_saved_path, map_location=device, weights_only=False)
    else:
        if os.path.exists(model_saved_path):
            print("Saved Hazard Transformer model uses the obsolete sigmoid head; retraining the PMF model.")
        model = ex_optuna(lambda trial: objective(trial, scenario_name))
        torch.save(model, model_saved_path)
    
    model.to(device)

    print("model summary")
    print(model)

    c_idx(model, DataLoader(HazardTransformerDataset(df_test, scenario_name), shuffle=True, collate_fn=custom_collate_fn, batch_size=256), df, device)
    auc(model, df, DataLoader(HazardTransformerDataset(df_test, scenario_name), shuffle=True, collate_fn=custom_collate_fn, batch_size=256), device)
    brier_score_evaluation(model, df, DataLoader(HazardTransformerDataset(df_test, scenario_name), shuffle=True, collate_fn=custom_collate_fn, batch_size=256), device)

if __name__ == '__main__':
    # run(ExperimentScenario.TIME_VARIANT)
    # run(ExperimentScenario.HETEROGENEOUS)
    # run(ExperimentScenario.EGFR_COMPONENTS)
    # run(ExperimentScenario.FIVELABMS)
    # Guard: skip if this rep's CKD_FIFTY_FEATURES_HETEROGENEOUS train data
    # doesn't exist yet (e.g. mid schema-migration, or a mini-experiment rep
    # that deliberately didn't build it) — otherwise get_train_test_data()
    # silently falls through to a full raw MIMIC extraction from
    # labevents.csv instead of erroring. See CLAUDE.md "Check a script's
    # actual entry point before running it as an experiment".
    if os.path.exists(ckd_fifty_features_heterogeneous_train_data_path):
        run(ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS)
    else:
        print(f"Skipping CKD_FIFTY_FEATURES_HETEROGENEOUS: no train data at {ckd_fifty_features_heterogeneous_train_data_path}")
    run(ExperimentScenario.FOUR_FEATURES)
    run(ExperimentScenario.EIGHT_FEATURES)
    run(ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS)
