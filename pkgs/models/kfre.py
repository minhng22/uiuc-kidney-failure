"""Closed-form Kidney Failure Risk Equation (KFRE) - 4-variable and 8-variable.

Not a trained model: risk is computed directly from the published Tangri et al. coefficients, so
there's no .fit() step and no *_model.pt/.dill artifact -- but the equation itself IS "the model"
here in every sense that matters for this codebase's model/experiment split (deepsurv/dynamicdeephit/
hazard_transformer/rnnsurv keep their architecture in pkgs/models/, their training/eval harness in
pkgs/experiments/; KFRE has no separate architecture-vs-training distinction since there's nothing to
train, so the equation lives here, wrapped in KFREModel for the same class-with-a-predictions()-method
shape every other model in this package has, and pkgs/experiments/kfre.py's run_kfre_model()/
caching/CLI entry point imports it).

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
import numpy as np
from pkgs.data_analysis.types import ExperimentScenario
from pkgs.data_analysis.model_data_store import get_last_observation_data

# S0(t): Original column, eAppendix 2, Tangri et al. 2016 JAMA - see module docstring.
S0_4VAR = {2: 0.9750, 5: 0.9240}
S0_8VAR = {2: 0.9780, 5: 0.9301}


def kfre_4var_risk(age, male, egfr, uacr, years=2):
    """Betas (-0.2201/0.2467/-0.5567/0.4510) and centering constants
    (7.036/0.5642/7.222/5.137) are the "Original" 4-variable Tangri et al.
    2011 (JAMA 305(15):1553-1559, PMID 21482743) derivation coefficients,
    as reprinted verbatim in eAppendix 2 of Tangri et al. 2016 (JAMA
    315(2):164-174, PMID 26757465) -- cross-checked against two
    independently-built open-source implementations (CRAN `kfre` R package,
    PyPI `kfre` Python package, both by the same domain-expert author) that
    reproduce the identical numbers. S0_4VAR values are eAppendix 2's
    "Original" column, North-American calibration (this repo's cohort is
    MIMIC-IV / US-based). Centering constants are each covariate's
    derivation-cohort reference value in the equation's own units (e.g.
    age/10 - 7.036 means the reference age is 70.36 years; ln(uACR) - 5.137
    means the reference uACR is exp(5.137) ~= 170 mg/g) -- confirmed against
    the Methods-text reference values reported in Tangri et al. 2016.
    Full citation trail: generated_data/rep1/kfre_8variable_coefficients_report.txt
    (that report covers the 8-variable model but the same eAppendix 2 source
    and cross-check method apply to the 4-variable numbers here) and
    EXPERIMENT_PLAN_DETAILS.md's "KFRE baseline model" section."""
    log_uacr = np.log(np.maximum(uacr, 1e-6))
    L = (-0.2201 * (age / 10 - 7.036)
         + 0.2467 * (male - 0.5642)
         - 0.5567 * (egfr / 5 - 7.222)
         + 0.4510 * (log_uacr - 5.137))
    return 1 - S0_4VAR[years] ** np.exp(L)


def kfre_8var_risk(age, male, egfr, uacr, albumin, phosphate, bicarbonate, calcium, years=2):
    """Betas (-0.1992/0.1602/-0.4919/0.3364/-0.3441/0.2604/-0.07354/-0.2228)
    and centering constants (7.036/0.5642/7.222/5.137/3.997/3.916/25.57/9.355)
    are the "Original" 8-variable Tangri et al. 2011 (JAMA 305(15):1553-1559,
    PMID 21482743) derivation coefficients, reprinted verbatim in Table 2 and
    eAppendix 2 of Tangri et al. 2016 (JAMA 315(2):164-174, PMID 26757465) --
    read directly from the publisher's own supplement PDF (not a third-party
    reimplementation), then cross-checked against two independently-built
    open-source packages that reproduce the identical numbers (see
    generated_data/rep1/kfre_8variable_coefficients_report.txt for the full
    trail, including the exact hazard-ratio table and PDF URL). "Original"
    (not the 2016 paper's separately-refit "Pooled" column) is used
    throughout, matching every clinical KFRE calculator and both open-source
    packages. S0_8VAR values are eAppendix 2's "Original" column,
    North-American calibration. Centering constants are each covariate's
    derivation-cohort reference value, confirmed against the Methods-text
    reference values Tangri et al. 2016 itself reports: "age, 70 years; 56%
    men; eGFR, 36 mL/min/1.73 m2; ACR, 170 mg/g; phosphate, 3.9 mg/dL;
    albumin, 4.0 g/dL; bicarbonate, 25.6 mEq/L; and calcium, 9.4 mg/dL" --
    matching age/10-7.036~=70, eGFR/5-7.222~=36, exp(5.137)~=170,
    phosphate-3.916~=3.9, albumin-3.997~=4.0, bicarbonate-25.57~=25.6,
    calcium-9.355~=9.4 to rounding."""
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


class KFREModel:
    """Unlike every other model in this package, there's no fitted state to
    hold -- KFRE is a closed-form equation, not a trained model -- so this
    class exists purely to give KFRE the same class-with-a-predictions()-
    method shape as every neural-net/lifelines/sksurv model here, per
    Stage 2.2's model-layer refactor. `scenario` (FOUR_FEATURES/
    EIGHT_FEATURES) is fixed at construction since it determines which
    equation (4- vs 8-variable) predictions() uses."""

    def __init__(self, scenario: ExperimentScenario):
        self.scenario = scenario

    def predictions(self, split='test'):
        """Unlike every other model in clinical_validity_analysis.py,
        KFRE's "risk score" IS already a genuine predicted probability of
        the event by a given number of years (1 - S0(t)^exp(L), see module
        docstring), not a generic score needing the calibrated-baseline-
        hazard fallback -- so both the 2yr and 5yr horizons (the exact
        years KFRE's S0 constants are defined for) are native, exact model
        output.

        Fetches get_last_observation_data(self.scenario)'s flattened
        one-row-per-patient frame itself -- scores each patient's last
        observation, not every raw lab-event row; see pkgs/models/cox.py's
        CoxModel.predictions docstring for why (cox/rnn_surv/kfre were the
        last 3 of 11 models still scored per row until Stage 2.2's fix).
        `split='train'` scores the training-set flattened frame instead
        (used to fit this model's own Breslow baseline hazard). Computed
        fresh via the closed-form equation each call (cheap, no training)
        rather than reading pkgs/experiments/kfre.py's cached
        <scenario>_kfre_<years>yr_risk_scores.csv, which is one row per RAW
        df_test row (the row-per-lab-event frame) and so isn't positionally
        aligned with this flattened frame. Returns
        (risk_scores, durations, events, native_prob_fn)."""
        df_train_flat, df_test_flat = get_last_observation_data(self.scenario)
        df_flat = df_train_flat if split == 'train' else df_test_flat
        risk_scores_2yr = compute_risk_scores(self.scenario, df_flat, years=2)
        risk_scores_5yr = compute_risk_scores(self.scenario, df_flat, years=5)

        def native_prob_fn(horizon_days):
            if round(horizon_days) == 730:
                return risk_scores_2yr
            if round(horizon_days) == 1825:
                return risk_scores_5yr
            return None

        return risk_scores_2yr, df_flat['duration_in_days'].values, df_flat['has_esrd'].values, native_prob_fn
