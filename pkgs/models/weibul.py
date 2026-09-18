"""WeibullAFTFitter (lifelines) has no custom nn.Module of its own -- a
direct call into the lifelines library, like pkgs/models/cox.py. WeibulModel
below gives it the same "model layer" home the neural-net models have.
"""
import numpy as np

from pkgs.data_analysis.types import ExperimentScenario
from pkgs.data_analysis.model_data_store import get_last_observation_data
from pkgs.experiments.utils import get_tv_rnn_model_features


class WeibulModel:
    def __init__(self, fitted_model):
        """`fitted_model` is the loaded lifelines WeibullAFTFitter (see
        pkgs/experiments/utils.py's load_pkl_and_dill_model -- that's what
        gets dilled to disk from training)."""
        self.fitted_model = fitted_model

    def predictions(self, scenario: ExperimentScenario, split='test'):
        """Fetches get_last_observation_data(scenario)'s flattened
        one-row-per-patient frame itself, restricted to this scenario's
        feature columns plus duration_in_days/has_esrd (see
        pkgs/experiments/weibul.py's run_scenario() for why
        duration_in_days==0 needs a small positive floor -- lifelines
        errors on duration==0). `split='train'` scores the training-set
        flattened frame instead (used to fit this model's own Breslow
        baseline hazard).

        WeibullAFTFitter predicts a median SURVIVAL TIME, not a risk
        score -- "higher=longer survival", the opposite of
        clinical_validity_analysis.py's convention. Negate it (same
        transform pkgs/experiments/weibul.py's own Brier-score call already
        uses: `-predicted_survival_times`) to get "higher=riskier". Returns
        (risk_scores, durations, events, native_prob_fn) -- see
        _native_prob_fn below."""
        df_train_flat, df_test_flat = get_last_observation_data(scenario)
        df_flat = df_train_flat if split == 'train' else df_test_flat
        features = get_tv_rnn_model_features(scenario)
        cols = features + ['duration_in_days', 'has_esrd']
        df_flat_selected = df_flat[cols].copy()
        df_flat_selected['duration_in_days'] = df_flat_selected['duration_in_days'].replace(0, 1e-5)

        predicted_survival_times = self.fitted_model.predict_median(df_flat_selected)
        risk_scores = (-predicted_survival_times.values if hasattr(predicted_survival_times, 'values')
                       else -np.asarray(predicted_survival_times))
        return (risk_scores, df_flat_selected['duration_in_days'].values,
                df_flat_selected['has_esrd'].values, self._native_prob_fn(df_flat_selected))

    def _native_prob_fn(self, df_flat_selected):
        """P(event by horizon_days) from WeibullAFTFitter's own parametric
        survival function, via `predict_survival_function(df, times=[t])`.

        PAPER_GAPS_EXPERIMENT_PLAN.md Gap 5b lists this wrapper among those
        returning None for native probabilities, noting that does not prove the
        estimator lacks a survival function. It does not: a Weibull AFT fit IS
        a closed-form survival function, S(t|x) = exp(-(t/lambda(x))^rho), and
        lifelines exposes it directly (confirmed against the fitted rep99
        artifacts). Unlike the non-parametric estimators, it is defined at every
        positive t, so there is no upper time bound to guard -- but the report
        still only quotes it inside the evaluation window where the cohort
        supports a comparison.

        Values are cached per horizon: the analyzer asks for ~50 grid points
        per Brier integration and each call is a full lifelines predict."""
        cache = {}

        def native_prob_fn(horizon_days):
            t = float(horizon_days)
            if t < 0:
                return None
            key = round(t, 6)
            if key not in cache:
                try:
                    surv = self.fitted_model.predict_survival_function(
                        df_flat_selected, times=[max(key, 1e-5)])
                except Exception:
                    return None
                cache[key] = 1.0 - np.asarray(surv.values, dtype=np.float64).reshape(-1)
            return cache[key]

        return native_prob_fn
