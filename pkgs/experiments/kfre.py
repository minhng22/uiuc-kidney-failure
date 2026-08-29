"""Closed-form Kidney Failure Risk Equation (KFRE) - 4-variable and 8-variable.

The equation itself (coefficients, S0 constants, kfre_4var_risk/kfre_8var_risk/compute_risk_scores)
lives in pkgs/models/kfre.py now -- see that module's docstring for the full citation trail and
the equations themselves. Not a trained model: risk is computed directly from the published Tangri
et al. coefficients, so there's no .fit() step and no *_model.pt/.dill artifact. This module is the
training/eval harness around that equation: caches the computed per-row risk scores
(<scenario>_kfre_<years>yr_risk_scores.csv under generated_data/rep<N>/, so repeat evaluation runs
don't recompute them) and reports the same C-index/Brier/AUC metrics as every other model here.
"""
import os

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

from pkgs.commons import generate_data_path_latest_rep
from pkgs.data_analysis.model_data_store import get_train_test_data
from pkgs.data_analysis.types import ExperimentScenario
from pkgs.experiments.utils import round_metric, compute_brier_score_from_risk_scores
from pkgs.models.kfre import compute_risk_scores


def get_kfre_risk_scores_path(scenario: ExperimentScenario, years=2):
    assert scenario in [ExperimentScenario.FOUR_FEATURES, ExperimentScenario.EIGHT_FEATURES]
    return f'{generate_data_path_latest_rep}/{scenario.value}_kfre_{years}yr_risk_scores.csv'


def run_kfre_model(scenario: ExperimentScenario, years=2):
    assert scenario in [ExperimentScenario.FOUR_FEATURES, ExperimentScenario.EIGHT_FEATURES], \
        f"KFRE has no published equation for {scenario} (only 4-/8-variable)"

    data_train, data_test = get_train_test_data(scenario)

    scores_path = get_kfre_risk_scores_path(scenario, years)
    if os.path.exists(scores_path):
        print(f"Using cached KFRE risk scores: {scores_path}")
        cached = pd.read_csv(scores_path)
        risk_scores_test = cached['risk_score'].values
    else:
        risk_scores_test = compute_risk_scores(scenario, data_test, years=years)
        pd.DataFrame({
            'subject_id': data_test['subject_id'].values,
            'risk_score': risk_scores_test,
        }).to_csv(scores_path, index=False)
        print(f"Cached KFRE risk scores: {scores_path}")

    print('Evaluate on test data')

    # Same sign convention as cox.py's run_cox_model: higher risk_scores_test means higher risk
    # (shorter survival); negate before passing to lifelines' concordance_index and the Brier-score
    # helper (both expect the opposite direction), but not before sksurv's cumulative_dynamic_auc
    # (which expects higher = higher risk directly).
    c_index_test = round_metric(concordance_index(data_test['duration_in_days'], -risk_scores_test, data_test['has_esrd']))
    print(f'Concordance Index Test: {c_index_test}')

    brier_score = compute_brier_score_from_risk_scores(data_train, data_test, -risk_scores_test)
    if brier_score is not None:
        print(f'Integrated Brier Score Test: {brier_score}')

    times = np.arange(1, 730, 1)
    y_train = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=data_train)
    y_test = Surv.from_dataframe(event='has_esrd', time='duration_in_days', data=data_test)
    _, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores_test, times)
    print(f"Mean time-dependent AUC: {mean_auc:.4f}")

    return c_index_test, brier_score, mean_auc


if __name__ == '__main__':
    print("\nRunning FOUR_FEATURES KFRE (4-variable) evaluation...")
    run_kfre_model(ExperimentScenario.FOUR_FEATURES)
    print("\nRunning EIGHT_FEATURES KFRE (8-variable) evaluation...")
    run_kfre_model(ExperimentScenario.EIGHT_FEATURES)
