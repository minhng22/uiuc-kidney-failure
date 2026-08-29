import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from pkgs.data_analysis.types import ExperimentScenario
from pkgs.data_analysis.model_data_store import get_train_test_data
from pkgs.experiments.utils import get_tv_rnn_model_features

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


class HazardTransformerDataset(Dataset):
    """Used for both training (pkgs/experiments/hazard_transformer.py's
    objective()/run()) and evaluation (this class's own predictions()
    method below) -- one dataset shape either way, since the model always
    consumes a subject's whole time-varying sequence regardless of use."""
    def __init__(self, df, scenario_name: ExperimentScenario):
        self.df = df
        self.subject_groups = list(df.groupby('subject_id'))

        self.scenario_name = scenario_name
        self.features = get_tv_rnn_model_features(scenario_name)

        self.max_seq_length = max(df.groupby('subject_id').size())

        # Cache per-column mean/std instead of recomputing them on every
        # __getitem__ call. Recomputing over the full df (previously
        # self.df[col].mean()/.std() inline below) is O(subjects x features x
        # N) per epoch — invisible at rep99's tiny mini-experiment scale but a
        # multi-hour stall at full Stage 3 scale (e.g. ~6.5M rows x 20 columns
        # x 26k subject accesses for TWENTY_FEATURES_HETEROGENEOUS).
        self._mean_cache = {}
        self._std_cache = {}

    def _mean(self, col):
        if col not in self._mean_cache:
            self._mean_cache[col] = self.df[col].mean()
        return self._mean_cache[col]

    def _std(self, col):
        if col not in self._std_cache:
            self._std_cache[col] = self.df[col].std()
        return self._std_cache[col]

    def __len__(self):
        return len(self.subject_groups)

    def __getitem__(self, idx):
        _, subject_data = self.subject_groups[idx]
        seq_length = len(subject_data)

        assert isinstance(subject_data, pd.DataFrame), f"subject_data is not a DataFrame: {type(subject_data)}"
        assert subject_data['duration_in_days'].is_monotonic_increasing, "subject_data is not sorted by time"

        features = np.zeros((self.max_seq_length, len(self.features)))
        mask = np.zeros(self.max_seq_length)

        if self.scenario_name == ExperimentScenario.TIME_VARIANT:
            features[:seq_length, 0] = (subject_data['egfr'].values - self._mean('egfr')) / self._std('egfr')
        elif self.scenario_name == ExperimentScenario.HETEROGENEOUS:
            features[:seq_length, 0] = (subject_data['egfr'].values - self._mean('egfr')) / self._std('egfr')
            features[:seq_length, 1] = subject_data['egfr_missing'].values
            features[:seq_length, 2] = (subject_data['protein'].values - self._mean('protein')) / self._std('protein')
            features[:seq_length, 3] = subject_data['protein_missing'].values
            features[:seq_length, 4] = (subject_data['albumin'].values - self._mean('albumin')) / self._std('albumin')
            features[:seq_length, 5] = subject_data['albumin_missing'].values
        elif self.scenario_name == ExperimentScenario.EGFR_COMPONENTS:
            features[:seq_length, 0] = (subject_data['age'].values - self._mean('age')) / self._std('age')
            features[:seq_length, 1] = subject_data['gender'].values
            features[:seq_length, 2] = (subject_data['serum_creatinine'].values - self._mean('serum_creatinine')) / self._std('serum_creatinine')
        elif self.scenario_name == ExperimentScenario.FIVELABMS:
            lab_names = ['egfr', 'hemoglobin']

            feature_idx = 0
            for lab in lab_names:
                features[:seq_length, feature_idx] = (subject_data[lab].values - self._mean(lab)) / self._std(lab)
                feature_idx += 1
                features[:seq_length, feature_idx] = subject_data[f'{lab}_missing'].values
                feature_idx += 1
        elif self.scenario_name == ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS:
            # 50 lab features with missingness indicators (100 features total)
            lab_names = ['egfr', 'urea_nitrogen', 'hemoglobin', 'serum_albumin', 'potassium',
                         'sodium', 'bicarbonate', 'phosphate', 'calcium', 'glucose',
                         'chloride', 'anion_gap', 'hematocrit', 'platelet_count', 'wbc',
                         'rbc', 'mcv', 'mch', 'mchc', 'rdw', 'magnesium', 'uric_acid',
                         'bilirubin_total', 'alt', 'ast', 'alkaline_phosphatase', 'ldh',
                         'iron', 'total_protein', 'cholesterol_total', 'triglycerides',
                         'inr', 'ptt', 'crp', 'ferritin', 'transferrin', 'tibc',
                         'lymphocytes', 'neutrophils', 'monocytes', 'basophils', 'eosinophils',
                         'pt', 'rdw_sd', 'lab_h', 'lab_l', 'lab_i',
                         'urine_specific_gravity', 'urine_ph', 'ph']
            feature_idx = 0
            for lab in lab_names:
                features[:seq_length, feature_idx] = (subject_data[lab].values - self._mean(lab)) / (self._std(lab) + 1e-8)
                feature_idx += 1
                features[:seq_length, feature_idx] = subject_data[f'{lab}_missing'].values
                feature_idx += 1
        elif self.scenario_name == ExperimentScenario.FOUR_FEATURES:
            features[:seq_length, 0] = (subject_data['age'].values - self._mean('age')) / self._std('age')
            features[:seq_length, 1] = subject_data['gender'].values
            features[:seq_length, 2] = (subject_data['egfr'].values - self._mean('egfr')) / self._std('egfr')
            features[:seq_length, 3] = (subject_data['uacr'].values - self._mean('uacr')) / self._std('uacr')
        elif self.scenario_name == ExperimentScenario.EIGHT_FEATURES:
            features[:seq_length, 0] = (subject_data['age'].values - self._mean('age')) / self._std('age')
            features[:seq_length, 1] = subject_data['gender'].values
            eight_feat_names = ['egfr', 'uacr', 'calcium', 'phosphate', 'bicarbonate', 'serum_albumin']
            for i, lab_name in enumerate(eight_feat_names):
                features[:seq_length, 2 + i] = (subject_data[lab_name].values - self._mean(lab_name)) / (self._std(lab_name) + 1e-8)
        elif self.scenario_name == ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS:
            # top 20 lab features with missingness indicators (40 features total)
            lab_names = ['egfr', 'potassium', 'urea_nitrogen', 'sodium', 'chloride', 'bicarbonate',
                         'anion_gap', 'hematocrit', 'platelet_count', 'hemoglobin', 'wbc', 'mchc',
                         'mch', 'rbc', 'mcv', 'rdw', 'glucose', 'calcium', 'magnesium', 'phosphate']
            feature_idx = 0
            for lab in lab_names:
                features[:seq_length, feature_idx] = (subject_data[lab].values - self._mean(lab)) / (self._std(lab) + 1e-8)
                feature_idx += 1
                features[:seq_length, feature_idx] = subject_data[f'{lab}_missing'].values
                feature_idx += 1

        mask[:seq_length] = 1

        time_to_event = subject_data['duration_in_days'].iloc[-1]
        event = np.array([subject_data['has_esrd'].iloc[-1]])

        return (torch.FloatTensor(features),
                torch.FloatTensor(mask),
                torch.LongTensor([time_to_event]),
                torch.FloatTensor(event),
                torch.FloatTensor(subject_data['duration_in_days'].values),
                torch.FloatTensor(subject_data['has_esrd'].values),
                torch.LongTensor([len(subject_data['duration_in_days'].values)]))


def custom_collate_fn(batch):
    features, masks, time_to_events, events, durations, esrds, _ = zip(*batch)

    features = pad_sequence(features, batch_first=True)
    masks = pad_sequence(masks, batch_first=True)
    durations = pad_sequence(durations, batch_first=True)
    esrds = pad_sequence(esrds, batch_first=True)

    return features, masks, torch.stack(time_to_events), torch.stack(events), durations, esrds


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
    cumulative incidence function), never as independent per-bin hazards.
    """

    ARCHITECTURE_VERSION = 2

    def __init__(self, input_dim, d_model, num_risks, num_layers, nhead, dropout, num_time_bins=100):
        super(HazardTransformer, self).__init__()
        # Instance marker makes old sigmoid-head checkpoints detectable.
        self.architecture_version = self.ARCHITECTURE_VERSION
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

    def predictions(self, scenario: ExperimentScenario, split='test'):
        """Used by pkgs/data_analysis/clinical_validity_analysis.py's Stage
        2.1/2.2 calibration + decision-curve report. Fetches the right
        train/test split itself and builds its own dataloader (this model
        always consumes a subject's WHOLE time-varying sequence, so it
        needs the raw multi-row frame from get_train_test_data(), not the
        one-row-per-patient flattened frame the sksurv-style models use)
        before doing the forward pass + PMF/CIF interpretation.

        HazardTransformer discretizes into NUM_TIME_BINS=100 bins spanning
        exactly [0, self.max_time=730] days (both hardcoded in __init__) —
        so cif[:, k] IS this model's genuine predicted P(event by
        ~k/99*730 days), not an arbitrary score. Read it off directly at any
        horizon <= 730; beyond that this model has no native prediction at
        all (it was never trained/evaluated past day 730), so
        native_prob_fn correctly returns None there and the caller falls
        back to extrapolation.

        Model output is a per-subject PMF over time bins (softmax-normalized
        — see this class's own docstring/forward()), not independent
        per-bin hazards — an earlier version used per-bin sigmoid hazards
        with survival = cumprod(1 - hazard), unbounded and prone to
        collapsing to predicting event probability 1.0 for every subject
        (see generated_data/rep1/ddh_collapse_fix_report.txt and this
        class's own docstring for the confirmed rep99 evidence). With a
        PMF, cumsum(pmf) IS the CIF directly. `split='train'` scores the
        training-set frame instead (used to fit this model's own Breslow
        baseline hazard). Returns
        (risk_scores, durations, events, native_prob_fn)."""
        if getattr(self, 'architecture_version', 1) != 2:
            raise ValueError(
                "This Hazard Transformer checkpoint uses the obsolete sigmoid head; retrain it before analysis."
            )
        df_train, df_test = get_train_test_data(scenario)
        df = df_train if split == 'train' else df_test
        dataloader = DataLoader(HazardTransformerDataset(df, scenario), shuffle=False,
                                 collate_fn=custom_collate_fn, batch_size=256)

        all_pmf, all_times, all_events = [], [], []
        self.eval()
        with torch.no_grad():
            for features, mask, time_to_events, event_indicators, _, _ in dataloader:
                pmf_preds, _, _ = self(features, mask)
                pmf = pmf_preds[:, 0, :].detach().cpu().numpy()
                all_pmf.append(pmf)
                all_times.extend(time_to_events.squeeze().numpy())
                all_events.extend(event_indicators.squeeze().numpy())
        pmf_matrix = np.concatenate(all_pmf, axis=0)
        cif_matrix = np.cumsum(pmf_matrix, axis=1)
        eval_times = np.linspace(0, 730, cif_matrix.shape[1])
        # Fixed common horizon (365d) for the scalar risk_scores used in
        # c-index/AUC/Brier -- NOT each patient's own observed time. That was
        # tried first (to dodge the final-bin-always-1.0 PMF degeneracy) but
        # introduced a worse confound: CIF only grows over time, so a patient
        # followed LONGER (typically lower-risk, especially censored) automatically
        # lands on a LATER bin with more accumulated risk regardless of true
        # risk -- confirmed empirically (eight_features/rep99: own-time c-index
        # 0.148, badly inverted; fixed-horizon c-index 0.477, a real if modest
        # reading). Same reasoning as pkgs/experiments/hazard_transformer.py's
        # own EVAL_HORIZON_DAYS -- 365 rather than the max_time=730 boundary,
        # which hits the OTHER degeneracy (last bin's CIF == 1.0 for every
        # softmax PMF). native_prob_fn below is unaffected by any of this -- it
        # already reads a fixed, caller-specified horizon per call, same as
        # every other model's own predictions() method.
        horizon_bin = int(round(min(365.0, float(self.max_time)) / float(self.max_time) * (cif_matrix.shape[1] - 1)))
        risk_scores = cif_matrix[:, horizon_bin]

        def native_prob_fn(horizon_days):
            if horizon_days > eval_times[-1]:
                return None
            idx = int(np.argmin(np.abs(eval_times - horizon_days)))
            return cif_matrix[:, idx]

        return risk_scores, np.array(all_times), np.array(all_events), native_prob_fn
