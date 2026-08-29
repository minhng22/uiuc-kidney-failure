import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

from pkgs.data_analysis.types import ExperimentScenario
from pkgs.data_analysis.model_data_store import get_train_test_data
from pkgs.experiments.utils import get_tv_rnn_model_features


class DynamicDeepHitDataset(Dataset):
    """Used for both training (pkgs/experiments/dynamic_deephit.py's
    objective()/run()) and evaluation (DynamicDeepHit.predictions() below)
    -- one dataset shape either way, since the model always consumes a
    subject's whole time-varying sequence regardless of use."""
    def __init__(self, df, scenario_name: ExperimentScenario):
        self.df = df
        self.subject_groups = list(df.groupby('subject_id'))

        self.scenario_name = scenario_name
        self.features = get_tv_rnn_model_features(scenario_name)

        self.max_seq_length = max(df.groupby('subject_id').size())

        # Cache per-column mean/std instead of recomputing them on every
        # __getitem__ call — see pkgs/models/hazard_transformer.py's
        # HazardTransformerDataset for the full rationale (multi-hour stall
        # at full Stage 3 scale).
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

    def number_of_subjects(self):
        return len(self.subject_groups)

    def __len__(self):
        return len(self.subject_groups)

    def get_all_subj_data(self):
        feats, masks, tte, ev, ttes, inds = [None for _ in range(6)]

        for i in range(len(self.subject_groups)):
            f_i, m_i, tte_i, ev_i, ttes_i, ind_i = self.__getitem__(i)
            if feats is None:
                feats = f_i.unsqueeze(0)
                masks = m_i.unsqueeze(0)
                tte = tte_i.unsqueeze(0)
                ev = ev_i.unsqueeze(0)
                ttes = ttes_i.unsqueeze(0)
                inds = ind_i.unsqueeze(0)
                print(f"feats shape: {feats.shape}")
                print(f"masks shape: {masks.shape}")
                print(f"tte shape: {tte.shape}")
            else:
                feats = torch.concat((feats, f_i.unsqueeze(0)), dim=0)
                masks = torch.concat((masks, m_i.unsqueeze(0)), dim=0)
                tte = torch.concat((tte, tte_i.unsqueeze(0)), dim=0)
                ev = torch.concat((ev, ev_i.unsqueeze(0)), dim=0)
                ttes = torch.concat((ttes, ttes_i.unsqueeze(0)), dim=0)
                inds = torch.concat((inds, ind_i.unsqueeze(0)), dim=0)

        print(f"feats shape: {feats.shape}")
        print(f"masks shape: {masks.shape}")
        print(f"tte shape: {tte.shape}")
        print(f"ev shape: {ev.shape}")
        print(f"ttes shape: {ttes.shape}")
        print(f"inds shape: {inds.shape}")

        return (
            feats, masks, tte, ev, ttes, inds
        )

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
            features[:seq_length, 0] = (subject_data['egfr'].values - self._mean('egfr')) / self._std('egfr')
            features[:seq_length, 1] = subject_data['egfr_missing'].values
            features[:seq_length, 2] = (subject_data['hemoglobin'].values - self._mean('hemoglobin')) / self._std('hemoglobin')
            features[:seq_length, 3] = subject_data['hemoglobin_missing'].values
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
            feat_idx = 0
            for lab_name in lab_names:
                features[:seq_length, feat_idx] = (subject_data[lab_name].values - self._mean(lab_name)) / (self._std(lab_name) + 1e-8)
                features[:seq_length, feat_idx + 1] = subject_data[f'{lab_name}_missing'].values
                feat_idx += 2
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
            feat_idx = 0
            for lab_name in lab_names:
                features[:seq_length, feat_idx] = (subject_data[lab_name].values - self._mean(lab_name)) / (self._std(lab_name) + 1e-8)
                features[:seq_length, feat_idx + 1] = subject_data[f'{lab_name}_missing'].values
                feat_idx += 2

        mask[:seq_length] = 1

        time_to_event = subject_data['duration_in_days'].iloc[-1]
        event = np.array([subject_data['has_esrd'].iloc[-1]])

        time_to_events = np.zeros((self.max_seq_length))
        time_to_events[:len(subject_data['duration_in_days'].values)] = subject_data['duration_in_days'].values

        event_indicators = np.zeros((self.max_seq_length))
        event_indicators[:len(subject_data['has_esrd'].values)] = subject_data['has_esrd'].values

        return (torch.FloatTensor(features),
                torch.FloatTensor(mask),
                torch.LongTensor([time_to_event]),
                torch.FloatTensor(event),
                torch.FloatTensor(time_to_events),
                torch.FloatTensor(event_indicators),
                torch.LongTensor([len(subject_data['has_esrd'].values)]))


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

    def predictions(self, scenario: ExperimentScenario, split='test'):
        """Used by pkgs/data_analysis/clinical_validity_analysis.py's Stage
        2.1/2.2 calibration + decision-curve report. Fetches the right
        train/test split itself and builds its own dataloader (this model
        always consumes a subject's WHOLE time-varying sequence, so it
        needs the raw multi-row frame from get_train_test_data(), not the
        one-row-per-patient flattened frame the sksurv-style models use)
        before doing the forward pass + PMF/CIF interpretation.
        `split='train'` scores the training-set frame instead (used to fit
        this model's own Breslow baseline hazard).

        Corrected from an earlier version that reused this class's own
        training-time brier_score_evaluation()/auc() row extraction as-is:
        that code paired hazard_preds[j][:p_seq_len] (the first p_seq_len
        entries of subject j's per-CALENDAR-DAY curve — see
        self.pred_times = 365*15, one value per literal day 0..5474) with
        time_to_events[j][:p_seq_len] (that subject's actual,
        irregularly-spaced LAB-VISIT days). Those two index sets don't
        correspond — "day 13" was being paired with "this patient's 13th
        lab visit" (e.g. day 1200), silently mixing calendar-day bins with
        visit-sequence position for every patient with non-daily visits
        (i.e. nearly all of them). Fixed here: model output is one curve
        per SUBJECT with a literal-day index (the model pools each
        subject's whole sequence via attention before predicting, see
        forward()), so it can be read off directly at any day t without
        needing to align it to that subject's own visit sequence at all.

        Model output is a per-subject PMF over time (softmax-normalized —
        see this class's own docstring/forward()), not independent per-day
        hazards: an earlier version of both this class and this method used
        per-day sigmoid hazards with survival = cumprod(1 - hazard), which
        had no bound on total predicted probability across time and could
        saturate to predicting event probability 1.0 for every subject
        regardless of input features (see
        generated_data/rep1/ddh_collapse_fix_report.txt). With a PMF,
        cumsum(pmf) IS the cumulative incidence function (CIF) directly —
        P(event by day t) = cumsum(pmf)[:, t] — no cumprod-of-survival
        needed. Operates per-subject via DynamicDeepHitDataset's internal
        grouping (one prediction per unique subject, not per raw dataframe
        row — this class only ever produces one curve per subject
        regardless of how many lab-event rows they have) — same evaluation
        unit every other model in clinical_validity_analysis.py uses (see
        pkgs/models/cox.py's CoxModel.predictions docstring for the
        history: cox/rnn_surv/kfre were the last 3 of 11 models still
        scored per row until Stage 2.2's fix). Returns
        (risk_scores, durations, events, native_prob_fn)."""
        df_train, df_test = get_train_test_data(scenario)
        df = df_train if split == 'train' else df_test
        dataloader = DataLoader(DynamicDeepHitDataset(df, scenario), shuffle=False, batch_size=16)

        all_pmf, all_durations, all_events = [], [], []
        self.eval()
        with torch.no_grad():
            for features, mask, time_to_event, event_indicator, time_to_events, event_indicators, seq_lens in dataloader:
                pmf_preds, _ = self(features, mask)
                pmf_preds = pmf_preds.cpu().detach().numpy()[:, 0, :]  # (batch, pred_times) per subject
                all_pmf.append(pmf_preds)
                all_durations.extend(time_to_event.squeeze(-1).cpu().numpy())
                all_events.extend(event_indicator.squeeze(-1).cpu().numpy())

        pmf_matrix = np.concatenate(all_pmf, axis=0)
        cif_matrix = np.cumsum(pmf_matrix, axis=1)  # P(event by day t)
        max_day_idx = cif_matrix.shape[1] - 1
        risk_scores = cif_matrix[:, min(730, max_day_idx)]
        durations = np.array(all_durations)
        events = np.array(all_events)

        def native_prob_fn(horizon_days):
            idx = int(round(horizon_days))
            if idx > max_day_idx:
                return None
            return cif_matrix[:, max(idx, 0)]

        return risk_scores, durations, events, native_prob_fn