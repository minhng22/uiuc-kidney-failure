"""LogisticHazard (pycox's LogisticHazard wrapping a plain MLPVanilla net) has
no custom nn.Module of its own in this codebase -- unlike deepsurv/
dynamicdeephit/hazard_transformer/rnnsurv, so there's no architecture class
already sitting in this package. LogisticHazardModel below gives it the
same "model layer" home those four have -- a class holding the wrapped
pycox model, with a predictions() method on it, same shape as
HazardTransformer.predictions()/DynamicDeepHit.predictions()/etc.
"""
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from pycox.models import LogisticHazard
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

from pkgs.data_analysis.types import ExperimentScenario
from pkgs.data_analysis.model_data_store import get_train_test_data
from pkgs.experiments.utils import get_tv_rnn_model_features


class LogisticHazardDataset(Dataset):
    """Used for both training (pkgs/experiments/logistic_hazard.py's
    objective()/run()) and evaluation (LogisticHazardModel.predictions()
    below, via prepare_data_for_pycox()) -- one dataset shape either way."""
    def __init__(self, df, scenario_name: ExperimentScenario):
        self.df = df
        self.subject_groups = list(df.groupby('subject_id'))
        self.scenario_name = scenario_name
        self.features = get_tv_rnn_model_features(scenario_name)

        # Cache per-column mean/std instead of recomputing them for every
        # subject in the loop below — see pkgs/models/hazard_transformer.py's
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

    def __len__(self):
        return len(self.subject_groups)

    def prepare_data_for_pycox(self):
        all_features = []
        all_durations = []
        all_events = []

        for _, subject_data in self.subject_groups:
            last_obs = subject_data.iloc[-1]

            if self.scenario_name == ExperimentScenario.TIME_VARIANT:
                features = [(last_obs['egfr'] - self._mean('egfr')) / self._std('egfr')]
            elif self.scenario_name == ExperimentScenario.HETEROGENEOUS:
                features = [
                    (last_obs['egfr'] - self._mean('egfr')) / self._std('egfr'),
                    last_obs['egfr_missing'],
                    (last_obs['protein'] - self._mean('protein')) / self._std('protein'),
                    last_obs['protein_missing'],
                    (last_obs['albumin'] - self._mean('albumin')) / self._std('albumin'),
                    last_obs['albumin_missing']
                ]
            elif self.scenario_name == ExperimentScenario.EGFR_COMPONENTS:
                features = [
                    (last_obs['age'] - self._mean('age')) / self._std('age'),
                    last_obs['gender'],
                    (last_obs['serum_creatinine'] - self._mean('serum_creatinine')) / self._std('serum_creatinine')
                ]
            elif self.scenario_name == ExperimentScenario.FIVELABMS:
                features = [
                    (last_obs['egfr'] - self._mean('egfr')) / self._std('egfr'),
                    last_obs['egfr_missing'],
                    (last_obs['hemoglobin'] - self._mean('hemoglobin')) / self._std('hemoglobin'),
                    last_obs['hemoglobin_missing'],
                ]
            elif self.scenario_name == ExperimentScenario.HETEROGENEOUS_IMPUTE:
                # Imputed heterogeneous: same features as FIVELABMS but without missingness indicators
                features = [
                    (last_obs['egfr'] - self._mean('egfr')) / self._std('egfr'),
                    (last_obs['hemoglobin'] - self._mean('hemoglobin')) / self._std('hemoglobin'),
                ]
            elif self.scenario_name == ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS:
                # 50 lab features with missingness indicators
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
                features = []
                for lab_name in lab_names:
                    features.append((last_obs[lab_name] - self._mean(lab_name)) / (self._std(lab_name) + 1e-8))
                    features.append(last_obs[f'{lab_name}_missing'])
            elif self.scenario_name == ExperimentScenario.FOUR_FEATURES:
                features = [
                    (last_obs['age'] - self._mean('age')) / self._std('age'),
                    last_obs['gender'],
                    (last_obs['egfr'] - self._mean('egfr')) / self._std('egfr'),
                    (last_obs['uacr'] - self._mean('uacr')) / self._std('uacr'),
                ]
            elif self.scenario_name == ExperimentScenario.EIGHT_FEATURES:
                features = [
                    (last_obs['age'] - self._mean('age')) / self._std('age'),
                    last_obs['gender'],
                ]
                for lab_name in ['egfr', 'uacr', 'calcium', 'phosphate', 'bicarbonate', 'serum_albumin']:
                    features.append((last_obs[lab_name] - self._mean(lab_name)) / (self._std(lab_name) + 1e-8))
            elif self.scenario_name == ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS:
                # top 20 lab features with missingness indicators
                lab_names = ['egfr', 'potassium', 'urea_nitrogen', 'sodium', 'chloride', 'bicarbonate',
                             'anion_gap', 'hematocrit', 'platelet_count', 'hemoglobin', 'wbc', 'mchc',
                             'mch', 'rbc', 'mcv', 'rdw', 'glucose', 'calcium', 'magnesium', 'phosphate']
                features = []
                for lab_name in lab_names:
                    features.append((last_obs[lab_name] - self._mean(lab_name)) / (self._std(lab_name) + 1e-8))
                    features.append(last_obs[f'{lab_name}_missing'])
            else:
                raise ValueError(f"Unsupported scenario: {self.scenario_name}")

            all_features.append(features)
            all_durations.append(last_obs['duration_in_days'])
            all_events.append(last_obs['has_esrd'])

        return np.array(all_features, dtype=np.float32), np.array(all_durations, dtype=np.float32), np.array(all_events, dtype=np.int32)


class LogisticHazardModel:
    def __init__(self, net):
        """`net` is the raw MLPVanilla loaded from disk (the .pt file saves
        just the net, not the pycox wrapper -- see
        pkgs/experiments/logistic_hazard.py's run()); wrapped into pycox's
        LogisticHazard here, the same way run() does, so callers pass the
        raw loaded net, not an already-wrapped model."""
        self.model = LogisticHazard(net, optimizer=optim.Adam(net.parameters()))

    def predictions(self, scenario: ExperimentScenario, split='test'):
        """Used by pkgs/data_analysis/clinical_validity_analysis.py's Stage
        2.1/2.2 calibration + decision-curve report. Fetches the right
        train/test split itself, builds its own one-row-per-patient inputs
        (LogisticHazardDataset already dedupes to one row per subject via
        prepare_data_for_pycox()'s `last_obs = subject_data.iloc[-1]`), and
        refits LabTransDiscreteTime(50) on the TRAINING split (regardless of
        `split`, since the bin boundaries must reflect the same fit used at
        training time -- this module's own run() has the same latent "raw
        bin index, not real days" issue this works around, see below).
        `split='train'` scores the training-set frame instead (used to fit
        this model's own Breslow baseline hazard).

        pycox's predict_surv_df gives a real survival curve — but its index
        is just the discrete bin POSITION (0..49 for LabTransDiscreteTime(50)),
        not a real day value (verified directly: surv.index ranges 0-49
        regardless of durations spanning thousands of days). An earlier
        version compared `horizon_days` (730/1825) against that raw index
        directly, which can never match — silently always fell back to the
        generic transform, every scenario, every horizon. Refitting
        LabTransDiscreteTime(50) on df_train recovers each bin's real day
        value via its `.cuts` array: quantile-based cuts are deterministic
        given the same input, so this reproduces the exact boundaries
        training used.

        Returns (risk_scores, durations, events, native_prob_fn)."""
        df_train, df_test = get_train_test_data(scenario)
        df = df_train if split == 'train' else df_test

        dataset = LogisticHazardDataset(df, scenario)
        x, durations, events = dataset.prepare_data_for_pycox()
        x = torch.tensor(x, dtype=torch.float32)

        labtrans = LabTransDiscreteTime(50)
        labtrans.fit_transform(df_train['duration_in_days'].values, df_train['has_esrd'].values)
        bin_real_days = labtrans.cuts  # bin_real_days[i] = real day value of surv.index i

        surv = self.model.predict_surv_df(x)  # index = discrete bin position, NOT real days

        median_time_idx = np.argmin(np.abs(bin_real_days - np.median(durations)))
        risk_scores = 1 - surv.iloc[median_time_idx].values

        def native_prob_fn(horizon_days):
            if horizon_days > bin_real_days.max():
                return None
            idx = int(np.argmin(np.abs(bin_real_days - horizon_days)))
            return 1.0 - surv.iloc[idx].values

        return risk_scores, durations, events, native_prob_fn
