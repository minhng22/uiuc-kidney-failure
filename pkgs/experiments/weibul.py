import os
import joblib
import numpy as np
from lifelines import WeibullAFTFitter
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
import dill

from pkgs.commons import (
    egfr_ti_weibul_model_path, four_features_weibul_model_path,
    eight_features_weibul_model_path, twenty_features_heterogeneous_weibul_model_path,
)
from pkgs.data_analysis.model_data_store import get_train_test_data, get_last_observation_data
from pkgs.data_analysis.types import ExperimentScenario
from pkgs.experiments.utils import load_pkl_and_dill_model, compute_brier_score_from_risk_scores, get_tv_rnn_model_features

# Path dict + scenario-aware run added for Stage 3's four/eight/
# twenty_features_heterogeneous scenarios, alongside (not replacing) the
# original NON_TIME_VARIANT-only run_ti() below. See gbsa.py's equivalent
# comment for why get_last_observation_data() is used; WeibullAFTFitter.fit()
# additionally uses every column of the dataframe passed to it as a
# covariate, so (unlike run_ti(), whose NON_TIME_VARIANT data is already
# just egfr/duration/event) the frame must be subset down to
# features + duration_in_days + has_esrd first -- passing subject_id/start/
# stop through untouched would fit nonsense covariates.
weibul_model_path_dict = {
    ExperimentScenario.FOUR_FEATURES: four_features_weibul_model_path,
    ExperimentScenario.EIGHT_FEATURES: eight_features_weibul_model_path,
    ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS: twenty_features_heterogeneous_weibul_model_path,
}

def compute_time_dependent_auc(model: WeibullAFTFitter, data_train, data_test, duration_col, event_col, times):
    y_train = Surv.from_dataframe(event=event_col, time=duration_col, data=data_train)
    y_test = Surv.from_dataframe(event=event_col, time=duration_col, data=data_test)
    cum_hazard = model.predict_cumulative_hazard(data_test, times=times).values
    cum_hazard = cum_hazard.T # shape of weibull model is (n_times, n_samples). shapes required for cumulative_dynamic_auc is (n_samples, n_times) 

    print(f"cumulative hazard shape: {cum_hazard.shape}")
    auc_values, mean_auc = cumulative_dynamic_auc(y_train, y_test, cum_hazard, times)

    return auc_values, mean_auc

def run_ti():
    df, df_test = get_train_test_data(ExperimentScenario.NON_TIME_VARIANT)

    df['duration_in_days'] = df['duration_in_days'].replace(0, 1e-5)
    df_test['duration_in_days'] = df_test['duration_in_days'].replace(0, 1e-5)

    print(f"Train data shape: {df.shape}")
    print(f"Test data shape: {df_test.shape}")

    trained_model = load_pkl_and_dill_model(egfr_ti_weibul_model_path)

    if not trained_model:
        model = WeibullAFTFitter()
        print('Fitting model:')
        model.fit(df, event_col='has_esrd', duration_col='duration_in_days')
        with open(egfr_ti_weibul_model_path, 'wb') as f:
            dill.dump(model, f, protocol=4)
    else:
        print("Loading model from file")
        model = trained_model

    print('Evaluate on test data')
    times = np.arange(1, 730, 1)

    predicted_survival_times = model.predict_median(df_test)
    event_occurred = df_test['has_esrd'].values
    actual_survival_times = df_test['duration_in_days'].values

    c_index = concordance_index(actual_survival_times, predicted_survival_times, event_occurred)
    print(f"C-index: {c_index:.4f}")

    # Compute Brier Score
    brier_score = compute_brier_score_from_risk_scores(df, df_test, -predicted_survival_times)  # Negative because higher survival time = lower risk
    if brier_score is not None:
        print(f'Integrated Brier Score Test: {brier_score}')

    _, mean_auc = compute_time_dependent_auc(model, df, df_test, 'duration_in_days', 'has_esrd', times)
    print(f"Mean time-dependent AUC: {mean_auc:.4f}")

def run_scenario(scenario: ExperimentScenario):
    """Scenario-aware entry point for four_features/eight_features/
    twenty_features_heterogeneous."""
    df, df_test = get_last_observation_data(scenario)
    features = get_tv_rnn_model_features(scenario)
    cols = features + ['duration_in_days', 'has_esrd']
    df = df[cols].copy()
    df_test = df_test[cols].copy()

    df['duration_in_days'] = df['duration_in_days'].replace(0, 1e-5)
    df_test['duration_in_days'] = df_test['duration_in_days'].replace(0, 1e-5)

    model_path = weibul_model_path_dict[scenario]
    trained_model = load_pkl_and_dill_model(model_path)

    if not trained_model:
        # penalizer=0.1: twenty_features_heterogeneous's flattened data carries
        # several low-variance binary *_missing indicator columns (e.g.
        # bicarbonate_missing) that WeibullAFTFitter's unregularized MLE hits
        # as near-complete separation on at rep1's full scale (26080 subjects)
        # -- reproduced via lifelines.exceptions.ConvergenceError, fixed by
        # adding a small penalizer per the library's own suggested remedy.
        # Verified against rep1's actual training data (read-only, no model
        # file written) before this fix was applied; rep99's smaller sample
        # never triggered this failure in the first place, so there was
        # nothing to re-verify there.
        model = WeibullAFTFitter(penalizer=0.1)
        print('Fitting model:')
        model.fit(df, event_col='has_esrd', duration_col='duration_in_days')
        with open(model_path, 'wb') as f:
            dill.dump(model, f, protocol=4)
    else:
        print("Loading model from file")
        model = trained_model

    print('Evaluate on test data')
    times = np.arange(1, min(730, int(df_test['duration_in_days'].max())), 1)

    predicted_survival_times = model.predict_median(df_test)
    event_occurred = df_test['has_esrd'].values
    actual_survival_times = df_test['duration_in_days'].values

    c_index = concordance_index(actual_survival_times, predicted_survival_times, event_occurred)
    print(f"C-index: {c_index:.4f}")

    brier_score = compute_brier_score_from_risk_scores(df, df_test, -predicted_survival_times)
    if brier_score is not None:
        print(f'Integrated Brier Score Test: {brier_score}')

    try:
        _, mean_auc = compute_time_dependent_auc(model, df, df_test, 'duration_in_days', 'has_esrd', times)
        print(f"Mean time-dependent AUC: {mean_auc:.4f}")
    except Exception as e:
        print(f"Warning: could not compute AUC: {e}")


if __name__ == '__main__':
    run_ti()
    print("\nRunning FOUR_FEATURES Weibull AFT model evaluation...")
    run_scenario(ExperimentScenario.FOUR_FEATURES)
    print("\nRunning EIGHT_FEATURES Weibull AFT model evaluation...")
    run_scenario(ExperimentScenario.EIGHT_FEATURES)
    print("\nRunning TWENTY_FEATURES_HETEROGENEOUS Weibull AFT model evaluation...")
    run_scenario(ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS)