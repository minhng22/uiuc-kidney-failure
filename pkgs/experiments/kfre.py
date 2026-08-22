"""Closed-form Kidney Failure Risk Equation (KFRE) - 4-variable and 8-variable.

Not a trained model: risk is computed directly from the published Tangri et al. coefficients, so
there's no .fit() step and no *_model.pt/.dill artifact. Only the computed per-row risk scores are
cached (<scenario>_kfre_<years>yr_risk_scores.csv under generated_data/rep<N>/), so repeat
evaluation runs don't recompute them.

Coefficients, centering constants, and S0 baseline-survival constants are all directly confirmed
against eAppendix 2 of Tangri N, Grams ME, Levey AS, et al. "Multinational assessment of accuracy
of equations for predicting risk of kidney failure: a meta-analysis." JAMA. 2016;315(2):164-174 -
fetched and read directly from the publisher's supplement PDF (not a third-party reimplementation).
The "Original" (not "Regional Calibrated" or "Pooled") row of that table is used throughout, which
is also what every clinical KFRE calculator and the CRAN/PyPI `kfre` packages implement - see
generated_data/rep1/kfre_8variable_coefficients_report.txt for the full citation trail, the
verbatim equations, and why "Original" (not the North-America-specific "Regional Calibrated" row)
is the correct choice here.

4-variable equation:
  L = -0.2201*(age/10 - 7.036) + 0.2467*(male - 0.5642) - 0.5567*(eGFR/5 - 7.222)
      + 0.4510*(ln(uACR) - 5.137)
  Risk(t) = 1 - S0(t)^exp(L),  S0(2yr) = 0.9750,  S0(5yr) = 0.9240

8-variable equation:
  L = -0.1992*(age/10 - 7.036) + 0.1602*(male - 0.5642) - 0.4919*(eGFR/5 - 7.222)
      + 0.3364*(ln(uACR) - 5.137) - 0.3441*(albumin - 3.997) + 0.2604*(phosphate - 3.916)
      - 0.07354*(bicarbonate - 25.57) - 0.2228*(calcium - 9.355)
  Risk(t) = 1 - S0(t)^exp(L),  S0(2yr) = 0.9780,  S0(5yr) = 0.9301
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

# S0(t): Original column, eAppendix 2, Tangri et al. 2016 JAMA - see module docstring.
S0_4VAR = {2: 0.9750, 5: 0.9240}
S0_8VAR = {2: 0.9780, 5: 0.9301}


def kfre_4var_risk(age, male, egfr, uacr, years=2):
    log_uacr = np.log(np.maximum(uacr, 1e-6))
    L = (-0.2201 * (age / 10 - 7.036)
         + 0.2467 * (male - 0.5642)
         - 0.5567 * (egfr / 5 - 7.222)
         + 0.4510 * (log_uacr - 5.137))
    return 1 - S0_4VAR[years] ** np.exp(L)


def kfre_8var_risk(age, male, egfr, uacr, albumin, phosphate, bicarbonate, calcium, years=2):
    log_uacr = np.log(np.maximum(uacr, 1e-6))
    L = (-0.1992 * (age / 10 - 7.036)
         + 0.1602 * (male - 0.5642)
         - 0.4919 * (egfr / 5 - 7.222)
         + 0.3364 * (log_uacr - 5.137)
         - 0.3441 * (albumin - 3.997)
         + 0.2604 * (phosphate - 3.916)
         - 0.07354 * (bicarbonate - 25.57)
         - 0.2228 * (calcium - 9.355))
    return 1 - S0_8VAR[years] ** np.exp(L)


def compute_risk_scores(scenario: ExperimentScenario, df, years=2):
    assert scenario in [ExperimentScenario.FOUR_FEATURES, ExperimentScenario.EIGHT_FEATURES]
    if scenario == ExperimentScenario.FOUR_FEATURES:
        return kfre_4var_risk(df['age'].values, df['gender'].values, df['egfr'].values,
                               df['uacr'].values, years=years)
    return kfre_8var_risk(df['age'].values, df['gender'].values, df['egfr'].values,
                           df['uacr'].values, df['serum_albumin'].values, df['phosphate'].values,
                           df['bicarbonate'].values, df['calcium'].values, years=years)


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
