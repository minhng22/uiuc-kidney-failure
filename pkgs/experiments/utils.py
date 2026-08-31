import numpy as np
from lifelines.utils import concordance_index
from sksurv.metrics import integrated_brier_score, brier_score
from sksurv.util import Surv
import torch
import optuna
from pkgs.data_analysis.types import ExperimentScenario
import os
import dill
import joblib

# from doc: "y must be a structured array with the first field being a binary class event indicator and the second field the time of the event/censoring"
def get_y_for_sckit_survival_model(df):
    return np.array(list(zip(df['has_esrd'].astype(bool), df['duration_in_days'])),
              dtype=[('event', bool), ('time', np.float64)])

# X must be a 2D array
def get_x_for_sckit_survival_model(df, scenario=None):
    """Default (scenario=None) preserves the original NON_TIME_VARIANT-only
    behavior (single 'egfr' feature). Pass a scenario to get that scenario's
    full feature list instead (via get_tv_rnn_model_features), e.g. for
    FOUR_FEATURES/EIGHT_FEATURES/TWENTY_FEATURES_HETEROGENEOUS."""
    if scenario is None:
        X = df['egfr'].values.reshape(-1, 1)
    else:
        X = df[get_tv_rnn_model_features(scenario)].values
    print(f'X shape: {X.shape}')
    return X


def round_metric(metric_num):
    return round(metric_num, 3)


def round_metric(metric_num):
    return round(metric_num, 3)


def compute_brier_score_from_survival_probs(df_train, df_test, survival_probs, times):
    """
    Compute Integrated Brier Score from survival probabilities.
    
    Args:
        df_train: Training dataframe with 'has_esrd' and 'duration_in_days' columns
        df_test: Test dataframe with 'has_esrd' and 'duration_in_days' columns  
        survival_probs: Array of survival probabilities of shape (n_samples, n_times)
        times: Array of time points corresponding to survival probabilities
    
    Returns:
        Integrated Brier Score
    """
    y_train = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df_train)
    y_test = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=df_test)
    
    try:
        # Ensure survival_probs has the right shape (n_samples, n_times)
        if survival_probs.ndim == 1:
            survival_probs = survival_probs.reshape(1, -1)
        elif survival_probs.shape[0] != len(df_test):
            survival_probs = survival_probs.T
            
        ibs = integrated_brier_score(y_train, y_test, survival_probs, times)
        return round_metric(ibs)
    except Exception as e:
        print(f"Warning: Could not compute Brier Score: {e}")
        return None


def compute_brier_score_from_risk_scores(df_train, df_test, risk_scores):
    """
    Compute Brier Score from risk scores by converting to survival probabilities.
    Uses a simple exponential survival model approximation.
    
    Args:
        df_train: Training dataframe
        df_test: Test dataframe  
        risk_scores: Risk scores from the model
    
    Returns:
        Integrated Brier Score
    """
    try:
        # Define time points for evaluation
        max_time = max(df_train['duration_in_days'].max(), df_test['duration_in_days'].max())
        times = np.linspace(1, min(max_time, 730), 50)  # Evaluate up to 2 years
        
        # Convert risk scores to survival probabilities using exponential model
        # S(t) = exp(-lambda * t), where lambda is proportional to risk score
        # Normalize risk scores to be positive
        risk_scores_norm = risk_scores - risk_scores.min() + 0.01
        
        # Create survival probability matrix
        survival_probs = np.zeros((len(df_test), len(times)))
        for i, t in enumerate(times):
            survival_probs[:, i] = np.exp(-risk_scores_norm * t / 365.0)  # Scale by year
            
        return compute_brier_score_from_survival_probs(df_train, df_test, survival_probs, times)
    except Exception as e:
        print(f"Warning: Could not compute Brier Score from risk scores: {e}")
        return None


def evaluate_ti_scikit_survival_model(df_test, risk_scores, surv_funcs, df_train):
    # Concordance Index on test data
    c_index_test = round_metric(concordance_index(df_test['duration_in_days'], risk_scores, df_test['has_esrd']))
    print(f'Concordance Index Test: {round_metric(c_index_test)}')
    
    # Brier score on test data
    times_test = np.linspace(0, df_test['duration_in_days'].max(), 100, endpoint=False)
    pred_surv_test = np.asarray([fn(times_test) for fn in surv_funcs])

    df_train['has_esrd'] = df_train['has_esrd'].astype(bool)
    df_test['has_esrd'] = df_test['has_esrd'].astype(bool)

    bs_test = integrated_brier_score(
        df_train[['has_esrd', 'duration_in_days']].to_records(index=False), 
        df_test[['has_esrd', 'duration_in_days']].to_records(index=False), pred_surv_test, times_test)
    print(f'Integrated Brier Score (Test): {round_metric(bs_test)}')
    
    # Also compute Brier Score using our new comprehensive function
    brier_score_new = compute_brier_score_from_survival_probs(df_train, df_test, pred_surv_test.T, times_test)
    if brier_score_new is not None:
        print(f'Integrated Brier Score (New): {brier_score_new}')

def c_idx_rnn_model(model, df_test, features):
    X_test = torch.tensor(df_test[features].values, dtype=torch.float32).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        test_risk_scores = model(X_test)
        test_risk_scores = test_risk_scores[:, -1, :]

    c_index = round_metric(concordance_index(df_test['duration_in_days'], test_risk_scores.squeeze().numpy(), df_test['has_esrd']))
    print("C-Index on Test Data:", c_index)

def ex_optuna(objective, n_trials=10):
    print("Running Optuna hyperparameter optimization")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print(f"Best hyperparameters: {study.best_params}")
    best_model = trial.user_attrs["model"]

    return best_model

def get_tv_rnn_model_features(scenario_name: ExperimentScenario):
    if scenario_name == ExperimentScenario.TIME_VARIANT:
        return ['egfr']
    elif scenario_name == ExperimentScenario.HETEROGENEOUS:
        return ['egfr', 'egfr_missing', 'protein', 'protein_missing', 'albumin', 'albumin_missing']
    elif scenario_name == ExperimentScenario.EGFR_COMPONENTS:
        return ['age', 'gender', 'serum_creatinine']
    elif scenario_name == ExperimentScenario.FIVELABMS:
        return ['egfr', 'egfr_missing', 'hemoglobin', 'hemoglobin_missing']
    elif scenario_name == ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS:
        return ['egfr', 'egfr_missing', 
                'urea_nitrogen', 'urea_nitrogen_missing',
                'hemoglobin', 'hemoglobin_missing',
                'serum_albumin', 'serum_albumin_missing',
                'potassium', 'potassium_missing',
                'sodium', 'sodium_missing',
                'bicarbonate', 'bicarbonate_missing',
                'phosphate', 'phosphate_missing',
                'calcium', 'calcium_missing',
                'glucose', 'glucose_missing',
                'chloride', 'chloride_missing',
                'anion_gap', 'anion_gap_missing',
                'hematocrit', 'hematocrit_missing',
                'platelet_count', 'platelet_count_missing',
                'wbc', 'wbc_missing',
                'rbc', 'rbc_missing',
                'mcv', 'mcv_missing',
                'mch', 'mch_missing',
                'mchc', 'mchc_missing',
                'rdw', 'rdw_missing',
                'magnesium', 'magnesium_missing',
                'uric_acid', 'uric_acid_missing',
                'bilirubin_total', 'bilirubin_total_missing',
                'alt', 'alt_missing',
                'ast', 'ast_missing',
                'alkaline_phosphatase', 'alkaline_phosphatase_missing',
                'ldh', 'ldh_missing',
                'iron', 'iron_missing',
                'total_protein', 'total_protein_missing',
                'cholesterol_total', 'cholesterol_total_missing',
                'triglycerides', 'triglycerides_missing',
                'inr', 'inr_missing',
                'ptt', 'ptt_missing',
                'crp', 'crp_missing',
                'ferritin', 'ferritin_missing',
                'transferrin', 'transferrin_missing',
                'tibc', 'tibc_missing',
                'lymphocytes', 'lymphocytes_missing',
                'neutrophils', 'neutrophils_missing',
                'monocytes', 'monocytes_missing',
                'basophils', 'basophils_missing',
                'eosinophils', 'eosinophils_missing',
                'pt', 'pt_missing',
                'rdw_sd', 'rdw_sd_missing',
                'lab_h', 'lab_h_missing',
                'lab_l', 'lab_l_missing',
                'lab_i', 'lab_i_missing',
                'urine_specific_gravity', 'urine_specific_gravity_missing',
                'urine_ph', 'urine_ph_missing',
                'ph', 'ph_missing']
    elif scenario_name == ExperimentScenario.NON_TIME_VARIANT:
        return ['egfr']
    elif scenario_name == ExperimentScenario.FOUR_FEATURES:
        return ['age', 'gender', 'egfr', 'uacr']
    elif scenario_name == ExperimentScenario.EIGHT_FEATURES:
        return ['age', 'gender', 'egfr', 'uacr', 'calcium', 'phosphate', 'bicarbonate', 'serum_albumin']
    elif scenario_name == ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS:
        return ['egfr', 'egfr_missing',
                'potassium', 'potassium_missing',
                'urea_nitrogen', 'urea_nitrogen_missing',
                'sodium', 'sodium_missing',
                'chloride', 'chloride_missing',
                'bicarbonate', 'bicarbonate_missing',
                'anion_gap', 'anion_gap_missing',
                'hematocrit', 'hematocrit_missing',
                'platelet_count', 'platelet_count_missing',
                'hemoglobin', 'hemoglobin_missing',
                'wbc', 'wbc_missing',
                'mchc', 'mchc_missing',
                'mch', 'mch_missing',
                'rbc', 'rbc_missing',
                'mcv', 'mcv_missing',
                'rdw', 'rdw_missing',
                'glucose', 'glucose_missing',
                'calcium', 'calcium_missing',
                'magnesium', 'magnesium_missing',
                'phosphate', 'phosphate_missing']

def combine_loss(hazard_preds, time_intervals, event_indicators, num_risks, w1=0.5, w2=0.1):
    # Vectorized rewrite of the original O(batch^2 x num_timepoints) nested-Python-loop
    # implementation (numerically verified to match to <1e-5 relative error).
    batch_size = hazard_preds.size(0)
    num_timepoints = hazard_preds.size(2)

    total_loss = 0

    raw_time = time_intervals[:, 0]

    for risk in range(num_risks):
        risk_hazard_preds = hazard_preds[:, risk, :]
        risk_event_indicators = event_indicators[:, risk]

        time_indices = raw_time.clamp(max=num_timepoints - 1).long()

        # Clamp before log(): risk_hazard_preds comes from torch.sigmoid(), which
        # is bounded in (0,1) mathematically but saturates to exactly 0.0/1.0 in
        # float32 once the pre-sigmoid logit's magnitude gets large (~88+) —
        # plausible here since Optuna searches learning_rate up to 1e-2 with no
        # gradient clipping, and heavily right-skewed inputs (e.g. uacr) can
        # produce large-magnitude logits for outlier patients. Once that
        # happens, log(0) = -inf poisons the loss and every gradient after it
        # (observed: finite decreasing loss for a few epochs, then permanent
        # `nan`, on full-scale rep1/rep2/rep3 four_features — not just the
        # rep99 mini-sample). eps chosen to be well inside float32 precision
        # without perturbing well-behaved (non-saturated) predictions.
        eps = 1e-7
        safe_hazard_preds = risk_hazard_preds.clamp(min=eps, max=1.0 - eps)

        event_log_prob = torch.log(safe_hazard_preds[torch.arange(batch_size), time_indices]) * risk_event_indicators

        # censor_log_prob[i] = sum_{k < t_i} log(1 - hazard[i, k]), via cumsum instead of a Python loop
        log1m_cumsum = torch.cumsum(torch.log(1 - safe_hazard_preds), dim=1)
        censor_idx = (time_indices - 1).clamp(min=0)
        censor_log_prob = log1m_cumsum[torch.arange(batch_size), censor_idx]
        censor_log_prob = torch.where(time_indices > 0, censor_log_prob, torch.zeros_like(censor_log_prob))
        censor_log_prob = censor_log_prob * (1 - risk_event_indicators)

        log_likelihood_loss = -torch.mean(event_log_prob + censor_log_prob)

        # F_matrix[j, i] = cumulative hazard of subject j up to subject i's event time t_i
        hazard_cumsum = torch.cumsum(risk_hazard_preds, dim=1)
        f_idx = (time_indices - 1).clamp(min=0)
        F_matrix = hazard_cumsum[:, f_idx]
        zero_mask = (time_indices == 0).unsqueeze(0)
        F_matrix = F_matrix.masked_fill(zero_mask, 0.0)
        F_i_own = torch.diagonal(F_matrix)

        diff = F_i_own.unsqueeze(1) - F_matrix.t()  # diff[i, j] = F_i - F_j
        pair_mask = (raw_time.unsqueeze(1) < raw_time.unsqueeze(0)) & risk_event_indicators.bool().unsqueeze(1)

        count = pair_mask.sum()
        if count > 0:
            # Clamp the exponent: with long sequences (rep99's mini sample has an
            # outlier patient at 5644 timesteps), cumulative hazard F_i/F_j can grow
            # large enough that -diff/w2 overflows exp() to inf, which turned to NaN
            # loss/gradients partway through training (observed: finite decreasing
            # loss through epoch 5, NaN from epoch 6 on, for dynamic_deephit on
            # rep99). 50 is far past where exp() saturates any meaningful ranking
            # signal, so this only affects the pathological tail, not normal-range
            # sequences.
            ranking_loss = (torch.exp((-diff / w2).clamp(max=50.0)) * pair_mask.float()).sum() / count
        else:
            ranking_loss = torch.zeros((), device=risk_hazard_preds.device)

        total_loss += log_likelihood_loss * w1 + ranking_loss * w2

    return total_loss / num_risks


def combine_loss_pmf(pmf_preds, time_intervals, event_indicators, num_risks, w1=0.5, w2=0.1):
    """Same log-likelihood + ranking-loss structure as combine_loss(), but for
    DynamicDeepHit's softmax-normalized PMF output (see
    pkgs/models/dynamicdeephit.py's class docstring) instead of independent
    per-day sigmoid hazards. Only dynamic_deephit.py's training loop should
    call this -- hazard_transformer.py still outputs per-day sigmoid hazards
    and must keep using combine_loss() (left untouched above).

    Two things are structurally different from combine_loss(), not just a
    substitution of variable names:
      1. Per-day PMF mass already IS P(T=t) directly (softmax normalizes
         across all days at once), so there's no need for combine_loss()'s
         separate "prod(1-hazard) before t_i" term for event subjects --
         that information is already baked into a single day's PMF value by
         construction, whereas independent per-day sigmoids need it added
         explicitly (that missing term for event subjects, in the original
         combine_loss(), was itself part of the hazard-collapse bug -- see
         generated_data/rep1/ddh_collapse_fix_report.txt).
      2. The ranking loss's cumulative-incidence proxy is cumsum(pmf), which
         is bounded in [0, 1] by construction (unlike combine_loss()'s
         cumsum(hazard), which is an unbounded sum of up to `pred_times`
         independent (0,1) values) -- this is what stops
         exp(-diff/w2) from being able to blow up during training.
    """
    batch_size = pmf_preds.size(0)
    num_timepoints = pmf_preds.size(2)
    total_loss = 0
    raw_time = time_intervals[:, 0]
    eps = 1e-7

    for risk in range(num_risks):
        risk_pmf = pmf_preds[:, risk, :]
        risk_event_indicators = event_indicators[:, risk]
        time_indices = raw_time.clamp(max=num_timepoints - 1).long()

        cif = torch.cumsum(risk_pmf, dim=1)  # CIF(t) = P(T <= t), true model CIF

        pmf_at_t = risk_pmf[torch.arange(batch_size), time_indices].clamp(min=eps)
        event_log_prob = torch.log(pmf_at_t) * risk_event_indicators

        surv_at_t = (1.0 - cif[torch.arange(batch_size), time_indices]).clamp(min=eps)
        censor_log_prob = torch.log(surv_at_t) * (1 - risk_event_indicators)

        log_likelihood_loss = -torch.mean(event_log_prob + censor_log_prob)

        f_idx = (time_indices - 1).clamp(min=0)
        F_matrix = cif[:, f_idx]
        zero_mask = (time_indices == 0).unsqueeze(0)
        F_matrix = F_matrix.masked_fill(zero_mask, 0.0)
        F_i_own = torch.diagonal(F_matrix)
        diff = F_i_own.unsqueeze(1) - F_matrix.t()
        pair_mask = (raw_time.unsqueeze(1) < raw_time.unsqueeze(0)) & risk_event_indicators.bool().unsqueeze(1)
        count = pair_mask.sum()
        if count > 0:
            ranking_loss = (torch.exp((-diff / w2).clamp(max=50.0)) * pair_mask.float()).sum() / count
        else:
            ranking_loss = torch.zeros((), device=pmf_preds.device)

        total_loss += log_likelihood_loss * w1 + ranking_loss * w2

    return total_loss / num_risks


def load_pkl_and_dill_model(model_path):
    model_pkl_path = model_path.replace('.dill', '.pkl')

    if not os.path.exists(model_path) and not os.path.exists(model_pkl_path):
        return None
    
    # some old models saved with .pkl extension
    if os.path.exists(model_pkl_path):
        print(f"Loading model from {model_pkl_path}")
        return joblib.load(model_pkl_path)
    
    print(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        return dill.load(f)

def get_device():
    """
    Picks a GPU with enough free memory headroom to avoid CUDA OOM, tie-broken
    by lowest current utilization.gpu%. Queried fresh via `nvidia-smi` on
    every call (no caching), so concurrent processes each get a reasonably
    load-balanced pick rather than colliding on a single fixed or
    randomly-chosen GPU. GPU 0 is excluded, matching this function's prior
    behavior (random.randint(1, 7)). Falls back to a random GPU in 1-7 if
    nvidia-smi is unavailable or its output can't be parsed, and to CPU if
    CUDA itself isn't available.

    Free memory is the primary sort key (not utilization) because a GPU can
    sit at 0% utilization while another tenant's process holds most of its
    memory allocated-but-idle (observed: rep4/rep5's rnnsurv and rep5's
    deepsurv CUDA-OOM'd on cuda:3, which had 0% utilization but only ~1GB
    free out of 49GB due to an unrelated user's job — see
    EXPERIMENT_STATUS.md Stage 3.1 rep4/rep5 notes). Candidates under
    MIN_FREE_MEM_MIB are dropped entirely when any GPU clears that bar, so a
    "just idle enough" but memory-starved GPU is never preferred over one
    with real headroom. Ties within TIE_BUCKET_MIB of each other are shuffled
    before the tie-break so concurrent processes launched together (e.g.
    `run_rep.sh`'s ~11 subprocesses) don't all deterministically pick the
    exact same "best" GPU and collide there a moment later.
    """
    import random
    import subprocess

    MIN_FREE_MEM_MIB = 4000  # skip GPUs with less headroom than this, if any GPU has more
    TIE_BUCKET_MIB = 5000  # candidates within this many MiB of each other are shuffled, not deterministically ordered

    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device("cpu")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        candidates = []
        for line in result.stdout.strip().splitlines():
            idx, util, mem_used, mem_total = (int(x.strip()) for x in line.split(","))
            if idx == 0:
                continue  # GPU 0 excluded, matching prior random.randint(1, 7) behavior
            free_mem = mem_total - mem_used
            candidates.append((free_mem, util, idx))
        if not candidates:
            raise RuntimeError("no candidate GPUs found in nvidia-smi output (all filtered out)")

        safe_candidates = [c for c in candidates if c[0] >= MIN_FREE_MEM_MIB]
        pool = safe_candidates if safe_candidates else candidates

        random.shuffle(pool)  # break thundering-herd ties before the stable sort below
        pool.sort(key=lambda c: (-c[0] // TIE_BUCKET_MIB, c[1]))  # bucket by free_mem (coarse), then lowest utilization within bucket
        best_free_mem, best_util, best_idx = pool[0]
        device = torch.device(f"cuda:{best_idx}")
        print(f"Using GPU: {device} (utilization={best_util}%, free_mem={best_free_mem}MiB)")
        return device
    except Exception as e:
        print(f"Warning: nvidia-smi-based GPU selection failed ({e}); falling back to random GPU 1-7")
        gpu_id = random.randint(1, 7)
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {device}")
        return device
    