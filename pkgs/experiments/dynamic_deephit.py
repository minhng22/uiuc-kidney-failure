import pandas as pd
from pkgs.commons import egfr_tv_dynamic_deep_hit_model_path, hg_dynamic_deep_hit_model_path, egfr_components_dynamic_deep_hit_model_path, fivelabms_dynamic_deep_hit_model_path, ckd_fifty_features_heterogeneous_dynamic_deep_hit_model_path, four_features_dynamic_deep_hit_model_path, eight_features_dynamic_deep_hit_model_path, twenty_features_heterogeneous_dynamic_deep_hit_model_path, ckd_fifty_features_heterogeneous_train_data_path
from pkgs.data_analysis.model_data_store import get_train_test_data
from pkgs.models.dynamicdeephit import DynamicDeepHit, DynamicDeepHitDataset
import torch
from torch.utils.data import DataLoader

from pkgs.experiments.utils import ex_optuna, get_tv_rnn_model_features, combine_loss_pmf, compute_brier_score_from_risk_scores
from pkgs.data_analysis.types import ExperimentScenario

import os
import numpy as np
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored
from pkgs.experiments.utils import get_device

num_risks = 1 # esrd
model_saved_path_dict = {
        ExperimentScenario.TIME_VARIANT: egfr_tv_dynamic_deep_hit_model_path,
        ExperimentScenario.HETEROGENEOUS: hg_dynamic_deep_hit_model_path,
        ExperimentScenario.EGFR_COMPONENTS: egfr_components_dynamic_deep_hit_model_path,
        ExperimentScenario.FIVELABMS: fivelabms_dynamic_deep_hit_model_path,
        ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS: ckd_fifty_features_heterogeneous_dynamic_deep_hit_model_path,
        ExperimentScenario.FOUR_FEATURES: four_features_dynamic_deep_hit_model_path,
        ExperimentScenario.EIGHT_FEATURES: eight_features_dynamic_deep_hit_model_path,
        ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS: twenty_features_heterogeneous_dynamic_deep_hit_model_path,
    }

def objective(trial, scenario_name: ExperimentScenario):
    device = get_device()

    print(f"Running trial {trial.number} for {scenario_name} on device {device}")
    df, _ = get_train_test_data(scenario_name)

    dataset = DynamicDeepHitDataset(df, scenario_name)
    # batch_size reduced from 256 -> 16: with this scenario's max patient
    # sequence length (5644 timesteps observed on rep99), batch_size=256
    # through the LSTM's backward graph exhausted a 10.5GB GPU
    # (torch.OutOfMemoryError trying to allocate 2MiB with 10.56GiB already
    # in use). 16 keeps memory well within a single 2080 Ti regardless of
    # which hidden_dim/num_layer Optuna trial is running.
    train_loader = DataLoader(dataset, shuffle=True, batch_size=16)

    input_dim = len(get_tv_rnn_model_features(scenario_name))
    num_layers = trial.suggest_int("num_layer", 1, 4)
    hidden_dims = [trial.suggest_int(f"hidden_dim_{i}", 16, 128) for i in range(num_layers)]
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    drop_out_lstm = trial.suggest_float('drop_out_rate', 0.1, 0.5)
    drop_out_cause = trial.suggest_float('drop_out_rate', 0.1, 0.5)
    llh_loss = trial.suggest_float('llh_loss', 0.1, 1.0)
    ranking_loss = 1 - llh_loss
    num_epochs = 50

    model = DynamicDeepHit(input_dim, hidden_dims, num_risks, drop_out_lstm, drop_out_cause).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping parameters
    patience = 5
    best_loss = float('inf')
    patience_counter = 0

    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        total_loss = 0
        for i, (features, mask, time_to_event, event_indicator, time_to_events, event_indicators, seq_lens) in enumerate(train_loader):
            debug_mode = False
            if i == 0:
                debug_mode = True
            features, mask, time_to_event, event_indicator = [x.to(device) for x in (features, mask, time_to_event, event_indicator)]
            optimizer.zero_grad()

            if debug_mode:
                print(f"features shape: {features.shape}, mask shape: {mask.shape}, time_to_event shape: {time_to_event.shape}, "
                      f"event_indicator shape: {event_indicator.shape}, time_to_events shape: {time_to_events.shape}, "
                      f"event_indicators shape: {event_indicators.shape}, sequence lengths shape: {seq_lens.shape}")

            pmf_preds, _ = model(features, mask, debug_mode)
            loss = combine_loss_pmf(pmf_preds, time_to_event, event_indicator, num_risks, llh_loss, ranking_loss)
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
    
    c_index = c_idx(model, dataset, device)

    saved_path = model_saved_path_dict[scenario_name]
    if os.path.exists(saved_path):
        saved_model = torch.load(saved_path, map_location=device)
        saved_cidx = c_idx(saved_model, dataset, device)
        
        print(f"Saved model C-index: {saved_cidx:.4f}, Current trial C-index: {c_index:.4f}")
        if c_index > saved_cidx:
            print("New model is better, saving current model")
            torch.save(model, saved_path)
        else:
            print("Saved model is better, keeping it")
    else:
        print("No existing model, saving current model")
        torch.save(model, saved_path)

    trial.set_user_attr(key="model", value=model)
    return c_index

def auc(model: DynamicDeepHit, test_dataset: DynamicDeepHitDataset, train_df: pd.DataFrame, device):
    y_train = Surv.from_arrays(
        event=train_df['has_esrd'].values, time=train_df['duration_in_days'].values, name_event='has_esrd', name_time='duration_in_days')

    # batch_size reduced 256 -> 16, same OOM reason as train_loader above;
    # this forward pass isn't wrapped in torch.no_grad() so it still builds
    # a full backward-capable graph.
    dataloader = DataLoader(test_dataset, shuffle=False, batch_size=16)

    # Accumulate across ALL batches before calling cumulative_dynamic_auc,
    # instead of calling it once per 16-patient mini-batch (as this used to
    # do). A small 16-patient batch has a much higher chance of having a
    # shorter max follow-up time than the full test set, so any batch whose
    # patients all happen to have short follow-up would hit sksurv's hard
    # "all times must be within follow-up time of test data" error against a
    # fixed 730-day grid — observed on full-scale rep2 (max batch follow-up
    # 346 days) even though cox.py's single whole-test-set AUC call, same
    # scenario/rep, didn't fail. Computing once over the full test set:
    # (a) matches cox.py's approach, (b) is what cumulative_dynamic_auc is
    # meant to be called with (per-batch calls were never statistically
    # correct AUCs to average in the first place), (c) removes the
    # mini-batch follow-up-range fragility entirely.
    all_time_to_events, all_risk_scores, all_event_indicators = [], [], []

    for i, (features, mask, time_to_event, event_indicator, time_to_events, event_indicators, seq_lens) in enumerate(dataloader):
        debug_mode = False
        # if i == 0:
        #     debug_mode = True
        features, mask, time_to_event, event_indicator, time_to_events, event_indicators, seq_lens = [x.to(device) for x in (features, mask, time_to_event, event_indicator, time_to_events, event_indicators, seq_lens)]

        if debug_mode:
            print(f"features shape: {features.shape}")
            print(f"mask shape: {mask.shape}")
            print(f"time_to_event shape: {time_to_event.shape}")
            print(f"time_to_events shape: {time_to_events.shape}")
            print(f"seq_lens shape: {seq_lens.shape}")

        pmf_preds, _ = model(features, mask, debug_mode)

        pmf_preds = pmf_preds.cpu().detach().numpy()
        pmf_preds = pmf_preds[:, 0, :] # only one risk, which is esrd
        # model output is a per-subject PMF over time (see DynamicDeepHit's
        # class docstring); cumsum gives the CIF (P(event by day t)), the
        # cumulative risk quantity AUC needs -- using the raw per-day PMF
        # value here would be an instantaneous, not cumulative, quantity.
        cif_preds = np.cumsum(pmf_preds, axis=1)

        if debug_mode:
            print(f"calc cif_preds shape: {cif_preds.shape}")

        for j in range(cif_preds.shape[0]):
            p_seq_len = int(seq_lens[j])
            all_time_to_events.append(time_to_events[j][:p_seq_len].cpu().detach().numpy())
            all_risk_scores.append(cif_preds[j][:p_seq_len])
            all_event_indicators.append(event_indicators[j][:p_seq_len].cpu().detach().numpy())

    f_time_to_events = np.concatenate(all_time_to_events, axis=0)
    f_risk_scores = np.concatenate(all_risk_scores, axis=0)
    f_event_indicators = np.concatenate(all_event_indicators, axis=0)

    print(f"f_time_to_events shape: {len(f_time_to_events)}")
    print(f"f_risk_scores shape: {len(f_risk_scores)}")
    print(f"f_event_indicators shape: {len(f_event_indicators)}")

    y_test = Surv.from_arrays(event=f_event_indicators, time=f_time_to_events, name_event='has_esrd', name_time='duration_in_days')

    # Bound times to the full test set's actual observed follow-up range
    # instead of a hardcoded 730-day (2yr) grid, so this can't fail even if
    # a particular rep/scenario's test set has shorter overall follow-up.
    max_time = min(y_test['duration_in_days'].max(), 729)
    times = np.arange(1, max(max_time, 2), 1)

    _, avg_auc = cumulative_dynamic_auc(y_train, y_test, f_risk_scores, times)
    print(f"Mean time-dependent AUC: {avg_auc:.2f}")

def brier_score_evaluation(model: DynamicDeepHit, test_dataset: DynamicDeepHitDataset, train_df: pd.DataFrame, device):
    """Compute Brier Score for Dynamic DeepHit model"""
    # batch_size reduced 256 -> 16, same OOM reason as train_loader above.
    dataloader = DataLoader(test_dataset, shuffle=False, batch_size=16)
    
    all_risk_scores = []
    all_times = []
    all_events = []
    
    for features, mask, time_to_event, event_indicator, time_to_events, event_indicators, seq_lens in dataloader:
        features, mask, time_to_event, event_indicator, time_to_events, event_indicators, seq_lens = [x.to(device) for x in (features, mask, time_to_event, event_indicator, time_to_events, event_indicators, seq_lens)]
        
        pmf_preds, _ = model(features, mask)
        pmf_preds = pmf_preds.cpu().detach().numpy()
        pmf_preds = pmf_preds[:, 0, :]  # only one risk (ESRD)
        # cumsum -> CIF, same reasoning as auc() above.
        cif_preds = np.cumsum(pmf_preds, axis=1)

        for j in range(cif_preds.shape[0]):
            p_seq_len = int(seq_lens[j])
            all_risk_scores.extend(cif_preds[j][:p_seq_len])
            all_times.extend(time_to_events[j][:p_seq_len].cpu().detach().numpy())
            all_events.extend(event_indicators[j][:p_seq_len].cpu().detach().numpy())
    
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

def simple_cindex(times, predictions, events):
    pairs = 0
    concordant = 0
    tied = 0
    
    print(f"Starting simple C-index calculation...")
    
    for i in range(len(times)):
        if not events[i]:
            continue
            
        for j in range(len(times)):
            if i == j:
                continue
                
            if times[j] > times[i]:
                pairs += 1
                if predictions[j] <= predictions[i]:
                    concordant += 1
    
    print(f"Pairs evaluated: {pairs}, Concordant: {concordant}")
    
    if pairs == 0:
        print("WARNING: No valid pairs found for comparison!")
        return 0.0
        
    return (concordant + tied) / pairs

def c_idx(model: DynamicDeepHit, dataset: DynamicDeepHitDataset, device, test=False):
    model.eval()

    # batch_size reduced 256 -> 16 for consistency with the other loaders
    # above (this one is under torch.no_grad() so was lower-risk, but keep
    # it aligned to be safe against the same seq_len=5644 outlier).
    loader = DataLoader(dataset, shuffle=False, batch_size=16)

    all_times = []
    all_events = []
    all_risks = []

    with torch.no_grad():
        debug_print = True
        for features, mask, T, E, _, _, _ in loader:
            features, mask = features.to(device), mask.to(device)
            pmf, _ = model(features, mask, False)
            pmf = pmf[:, 0, :].cpu().numpy()
            # model output is a per-subject PMF over time (see
            # DynamicDeepHit's class docstring) -- cumsum gives the
            # cumulative incidence function (CIF), i.e. P(event by day t),
            # which is the risk quantity c-index/AUC/Brier need. Using the
            # raw per-day PMF value here (as this used to, back when the
            # model output independent per-day hazards) would compare an
            # instantaneous quantity across subjects with different t_j
            # instead of a cumulative one.
            cif = np.cumsum(pmf, axis=1)

            if debug_print:
                print(f"features shape: {features.shape}, mask shape: {mask.shape}, T shape: {T.shape}, E shape: {E.shape} cif shape: {cif.shape}")

            T = T.cpu().numpy().ravel().astype(int)
            E = E.cpu().numpy().ravel().astype(bool)

            for j, t_j in enumerate(T):
                if debug_print:
                    print(f"Processing subject {j} with time {t_j} and event {E[j]} cif at t_j: {cif[j, t_j : t_j + 1]}")
                    debug_print = False
                all_times.append(t_j)
                all_events.append(E[j])
                all_risks.append(cif[j, t_j : t_j + 1])

    all_times = np.array(all_times)
    all_events = np.array(all_events)
    all_risks = np.array(all_risks).flatten()
    all_surv = 1.0 - all_risks

    print(f"all_E shape: {all_events.shape} first 10 values: {all_events[:10]}")
    print(f"all_R shape: {all_risks.shape} first 10 values: {all_risks[:10]}")
    print(f"all_T shape: {all_times.shape} first 10 values: {all_times[:10]}")

    cindex = concordance_index(all_times, all_surv, all_events)

    res = concordance_index_censored(all_events, all_times, all_risks)
    print(f"Global test C-index (scikit-survival): {res[0]:.3f} number of concordant pairs: {res[1]}")
    
    print(f"Global test C-index: {cindex:.3f}")

    if test:
        num_events = sum(all_events)
        print(f"Event rate: {num_events}/{len(all_events)} ({num_events/len(all_events)*100:.2f}%)")
        
        simple_ci = simple_cindex(all_times, all_risks, all_events)
        print(f"Simple C-index implementation: {simple_ci:.3f}")
    return cindex

# Update the run function to use the device
def run(scenario_name: ExperimentScenario):
    torch.backends.cudnn.enabled = False
    device = get_device()
    df, df_test = get_train_test_data(scenario_name)

    model_saved_path = model_saved_path_dict[scenario_name]

    if os.path.exists(model_saved_path):
        print("Loading from saved weights")
        model = torch.load(model_saved_path, map_location=device, weights_only = False)
    else:
        model = ex_optuna(lambda trial: objective(trial, scenario_name))
        torch.save(model, model_saved_path)

    model.to(device)
    print("model summary:")
    print(model)

    test_dataset = DynamicDeepHitDataset(df_test, scenario_name)

    c_idx(model, test_dataset, device, True)
    auc(model, test_dataset, df, device)
    brier_score_evaluation(model, test_dataset, df, device)

if __name__ == '__main__':
    # Per-scenario isolation: an uncaught exception in one scenario used to
    # abort the whole script, silently skipping every scenario after it
    # (observed at full Stage 3 scale: a four_features failure meant
    # eight_features/twenty_features_heterogeneous never even started, for
    # every rep that hit it). Catch and continue instead, mirroring
    # run_rep.sh's own per-experiment failure tolerance.
    for scenario in (ExperimentScenario.FOUR_FEATURES, ExperimentScenario.EIGHT_FEATURES, ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS):
        try:
            run(scenario)
        except Exception:
            import traceback
            print(f"✗ dynamic_deephit/{scenario.value} failed:")
            traceback.print_exc()