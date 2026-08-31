import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index

from pkgs.models.rnnsurv import RNNSurv
from pkgs.data_analysis.model_data_store import get_train_test_data
from pkgs.experiments.utils import (round_metric, ex_optuna,
                                    get_tv_rnn_model_features,
                                    compute_brier_score_from_survival_probs)
from pkgs.commons import (egfr_tv_rnn_surv_model_path, hg_rnn_surv_model_path, egfr_components_rnn_surv_model_path,
                          fivelabms_rnn_surv_model_path, ckd_fifty_features_heterogeneous_rnn_surv_model_path, current_rep,
                          four_features_rnn_surv_model_path, eight_features_rnn_surv_model_path,
                          twenty_features_heterogeneous_rnn_surv_model_path,
                          ckd_fifty_features_heterogeneous_train_data_path)
from pkgs.data_analysis.types import ExperimentScenario
from sksurv.metrics import cumulative_dynamic_auc
import numpy as np
from pkgs.experiments.utils import get_device

import os
from sksurv.util import Surv

class RNNSurvDataset(Dataset):
    def __init__(self, df, features, duration_col, event_col):
        self.X = torch.tensor(df[features].values, dtype=torch.float32).unsqueeze(1)
        self.durations = torch.tensor(df[duration_col].values, dtype=torch.float32)
        self.events = torch.tensor(df[event_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.durations[idx], self.events[idx]

def rnn_surv_loss(event_pmf, durations, events, time_intervals,
                  likelihood_loss_weight, eps=1e-7):
    """PMF likelihood plus a bounded cumulative-incidence ranking loss.

    Events are scored by their observed interval's mass; censored patients
    are scored by the probability of surviving beyond that interval.  The
    ranking term uses CIFs, which are bounded in [0, 1], unlike the old sum
    of independent sigmoid values.
    """
    pmf = event_pmf[:, -1, :]
    n_patients, n_time_intervals = pmf.shape
    interval_indices = torch.bucketize(durations, time_intervals, right=True)
    interval_indices = interval_indices.clamp(max=n_time_intervals - 1)
    patient_indices = torch.arange(n_patients, device=pmf.device)

    cif = torch.cumsum(pmf, dim=1)
    event_log_prob = torch.log(pmf[patient_indices, interval_indices].clamp(min=eps)) * events
    censor_log_prob = torch.log(
        (1.0 - cif[patient_indices, interval_indices]).clamp(min=eps)
    ) * (1.0 - events)
    likelihood_loss = -(event_log_prob + censor_log_prob).mean()

    # For an event i, compare its CIF with every patient j at i's observed
    # interval. Earlier failures should have higher cumulative risk.
    cif_at_event_times = cif[:, interval_indices]
    own_cif = torch.diagonal(cif_at_event_times)
    diff = own_cif.unsqueeze(1) - cif_at_event_times.T
    comparable_pairs = (
        (durations.unsqueeze(1) < durations.unsqueeze(0))
        & events.bool().unsqueeze(1)
    )
    pair_count = comparable_pairs.sum()
    if pair_count > 0:
        ranking_loss = (
            torch.exp((-diff / 0.1).clamp(max=50.0)) * comparable_pairs.float()
        ).sum() / pair_count
    else:
        ranking_loss = torch.zeros((), device=pmf.device)

    return likelihood_loss_weight * likelihood_loss + (1 - likelihood_loss_weight) * ranking_loss

def objective(trial, scenario_name: ExperimentScenario):
    device = get_device()

    print(f"Running trial number {trial.number} for {scenario_name} on device {device}")
    duration_col = 'duration_in_days'
    event_col = 'has_esrd'
    num_time_intervals = trial.suggest_int('num_time_intervals', 10, 50)
    rnn_surv_features = get_tv_rnn_model_features(scenario_name)

    df, _ = get_train_test_data(scenario_name)
    
    model_path_dict = {
        ExperimentScenario.TIME_VARIANT: egfr_tv_rnn_surv_model_path,
        ExperimentScenario.HETEROGENEOUS: hg_rnn_surv_model_path,
        ExperimentScenario.EGFR_COMPONENTS: egfr_components_rnn_surv_model_path,
        ExperimentScenario.FIVELABMS: fivelabms_rnn_surv_model_path,
        ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS: ckd_fifty_features_heterogeneous_rnn_surv_model_path,
        ExperimentScenario.FOUR_FEATURES: four_features_rnn_surv_model_path,
        ExperimentScenario.EIGHT_FEATURES: eight_features_rnn_surv_model_path,
        ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS: twenty_features_heterogeneous_rnn_surv_model_path,
    }
    model_saved_path = model_path_dict[scenario_name]

    train_dataset = RNNSurvDataset(df, rnn_surv_features, duration_col, event_col)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    input_dim = len(rnn_surv_features)
    embedding_size = trial.suggest_int('embedding_size', 32, 128)
    num_embedding_layers = trial.suggest_int('num_embedding_layers', 1, 3)
    hidden_dims = trial.suggest_int('hidden_dims', 64, 256)
    num_recurrent_layers = trial.suggest_int('num_recurrent_layers', 1, 3)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    cross_entropy_loss_weight = trial.suggest_float('cross_entropy_loss_weight', 0.1, 0.9)

    num_epochs = 50

    # Define time intervals based on the training data
    max_duration = df[duration_col].max()
    time_intervals = torch.linspace(0, max_duration, num_time_intervals + 1)[1:].to(device)

    model = RNNSurv(
        input_dim, embedding_size, num_embedding_layers, hidden_dims,
        num_recurrent_layers, num_time_intervals, max_time=max_duration,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping parameters
    patience = 5
    best_loss = float('inf')
    patience_counter = 0

    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        total_loss = 0
        for batch in train_loader:
            X_batch, durations_batch, events_batch = [x.to(device) for x in batch]
            optimizer.zero_grad()
            event_pmf, _ = model(X_batch)
            loss = rnn_surv_loss(
                event_pmf, durations_batch, events_batch, time_intervals,
                cross_entropy_loss_weight,
            )
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
    
    saved_model_is_compatible = False
    if os.path.exists(model_saved_path):
        saved_model = torch.load(model_saved_path, map_location=device, weights_only=False)
        saved_model_is_compatible = (
            getattr(saved_model, 'architecture_version', 1) == RNNSurv.ARCHITECTURE_VERSION
        )

    if saved_model_is_compatible:
        c_index = score_model_train(model, df, rnn_surv_features, device)
        saved_c_index = score_model_train(saved_model, df, rnn_surv_features, device)
        
        print(f"Current trial C-index: {c_index:.4f}, Previously saved model C-index: {saved_c_index:.4f}")
        if c_index > saved_c_index:
            print("Current model performs better, saving it...")
            torch.save(model, model_saved_path)
    else:
        print("No compatible saved model found, saving current PMF model")
        torch.save(model, model_saved_path)

    c_index = score_model_train(model, df, rnn_surv_features, device)
    
    trial.set_user_attr(key="model", value=model)
    return c_index

def _batched_model_forward(model: RNNSurv, X, batch_size=8192):
    """Runs model(X) in chunks instead of one unbounded forward pass.

    `df`/`df_test` here are per-lab-event rows, not per-patient -- for
    TWENTY_FEATURES_HETEROGENEOUS that's up to ~6.5M train / ~1.6M test rows
    (vs. thousands for four/eight_features). A single-shot forward pass over
    all of them at once (the code this replaces) makes the LSTM allocate
    workspace proportional to the whole batch and reliably CUDA-OOMs
    regardless of which GPU is picked or how much free memory it has
    (observed: ~29GB already resident, then a 95GB single allocation
    request against a 47GB GPU) -- see EXPERIMENT_STATUS.md Stage 3.1
    rep4/rep5 notes. Batching keeps peak memory bounded by batch_size
    instead of dataset size; numerically identical to the unbatched call.
    """
    event_pmf_chunks = []
    risk_chunks = []
    with torch.no_grad():
        for start in range(0, X.size(0), batch_size):
            pmf_chunk, risk_chunk = model(X[start:start + batch_size])
            event_pmf_chunks.append(pmf_chunk)
            risk_chunks.append(risk_chunk)
    return torch.cat(event_pmf_chunks, dim=0), torch.cat(risk_chunks, dim=0)

def _batched_risk_by_time(cif: torch.Tensor, bin_indices: torch.Tensor, batch_size=8192):
    """Computes cif[:, bin_indices].cpu().numpy() in row-chunks instead of
    one unbounded fancy-index + transfer.

    `cif` is [N_test, num_bins] on GPU; for TWENTY_FEATURES_HETEROGENEOUS,
    N_test is ~1.6M rows and `bin_indices` can span the full ~4200-day
    follow-up range, so the single-shot slice materializes a dense
    [N_test, len(times)] float32 tensor -- ~28GB observed, which exceeds
    this host's 10.57GB-per-GPU capacity outright (not a contention issue:
    it can never fit, regardless of how idle/free the GPU is). See
    EXPERIMENT_STATUS.md Stage 3.1 rep2/rep3 rnnsurv notes. Batching over
    N_test keeps peak GPU memory bounded by batch_size instead of dataset
    size; numerically identical to the unbatched call.
    """
    chunks = []
    with torch.no_grad():
        for start in range(0, cif.size(0), batch_size):
            chunks.append(cif[start:start + batch_size][:, bin_indices].cpu().numpy())
    return np.concatenate(chunks, axis=0)


def score_model_train(model: RNNSurv, df, features, device):
    X_test = torch.tensor(df[features].values, dtype=torch.float32).unsqueeze(1).to(device)
    model.eval()
    _, test_risk_scores = _batched_model_forward(model, X_test)
    test_risk_scores = test_risk_scores.squeeze()

    c_index = round_metric(concordance_index(df['duration_in_days'], -test_risk_scores.cpu().numpy(), df['has_esrd']))
    print("C-Index on Test Data:", c_index)

    return c_index

# Update the run function to use the device
def run(scenario_name: ExperimentScenario):
    device = get_device()
    df, df_test = get_train_test_data(scenario_name)

    model_path_dict = {
        ExperimentScenario.TIME_VARIANT: egfr_tv_rnn_surv_model_path,
        ExperimentScenario.HETEROGENEOUS: hg_rnn_surv_model_path,
        ExperimentScenario.EGFR_COMPONENTS: egfr_components_rnn_surv_model_path,
        ExperimentScenario.FIVELABMS: fivelabms_rnn_surv_model_path,
        ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS: ckd_fifty_features_heterogeneous_rnn_surv_model_path,
        ExperimentScenario.FOUR_FEATURES: four_features_rnn_surv_model_path,
        ExperimentScenario.EIGHT_FEATURES: eight_features_rnn_surv_model_path,
        ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS: twenty_features_heterogeneous_rnn_surv_model_path,
    }
    model_saved_path = model_path_dict[scenario_name]

    if os.path.exists(model_saved_path) and getattr(torch.load(model_saved_path, map_location='cpu', weights_only=False), 'architecture_version', 1) == RNNSurv.ARCHITECTURE_VERSION:
        print("Loading from saved weights")
        model = torch.load(model_saved_path, map_location=device, weights_only=False)
    else:
        if os.path.exists(model_saved_path):
            print("Saved RNN-Surv model uses the obsolete sigmoid head; retraining the PMF model.")
        model = ex_optuna(lambda trial: objective(trial, scenario_name))
    model.to(device)

    X_test = torch.tensor(df_test[get_tv_rnn_model_features(scenario_name)].values, dtype=torch.float32).unsqueeze(1).to(device)
    model.eval()
    event_pmf, test_risk_scores = _batched_model_forward(model, X_test)
    test_risk_scores = test_risk_scores.squeeze()

    c_index = round_metric(concordance_index(df_test['duration_in_days'], -test_risk_scores.cpu().numpy(), df_test['has_esrd']))
    print("C-Index on Test Data:", c_index)

    # PMF -> CIF -> survival curve, rather than fabricating a curve from the
    # scalar ranking score.  Evaluation times must stay within the test set's
    # observed follow-up range.
    max_eval_time = min(int(df_test['duration_in_days'].max()) - 1, int(model.max_time))
    times = np.arange(1, max_eval_time + 1)
    if len(times) == 0:
        print("Skipping Brier/AUC: no positive evaluation time within follow-up.")
        return

    cif = torch.cumsum(event_pmf[:, -1, :], dim=1)
    bin_indices = torch.clamp(
        (torch.as_tensor(times, device=device, dtype=torch.float32) / model.max_time
         * (cif.size(1) - 1)).round().long(),
        min=0, max=cif.size(1) - 1,
    )
    risk_by_time = _batched_risk_by_time(cif, bin_indices)
    survival_by_time = 1.0 - risk_by_time

    brier_score = compute_brier_score_from_survival_probs(
        df, df_test, survival_by_time, times,
    )
    if brier_score is not None:
        print(f'Integrated Brier Score Test: {brier_score}')

    y_train = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df)
    y_test = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df_test)
    # sksurv's cumulative_dynamic_auc internally does an O(N_test x
    # len(times)) argsort (and similarly-shaped intermediates) over the
    # full risk_by_time matrix -- for TWENTY_FEATURES_HETEROGENEOUS
    # (N_test ~1.6M, len(times) up to ~4700 days) this reliably exhausts
    # host memory (observed: numpy MemoryError allocating 28-56GB) even
    # though C-Index and Brier Score above succeed fine (those don't need
    # the full [N_test, len(times)] matrix). Same class of failure
    # independently hit in cox.py's TWENTY_FEATURES_HETEROGENEOUS eval --
    # see EXPERIMENT_STATUS.md Stage 3.1 rep2/rep3 notes. Mirrors the
    # existing try/except around this same sksurv call already in
    # srf.py's run_scenario().
    try:
        _, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_by_time, times)
    except MemoryError as e:
        print(f"Warning: could not compute AUC: {e}")
        mean_auc = None

    if mean_auc is not None:
        print(f"Mean time-dependent AUC: {mean_auc:.4f}")

if __name__ == '__main__':
    run(ExperimentScenario.FOUR_FEATURES)
    run(ExperimentScenario.EIGHT_FEATURES)
    run(ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS)
