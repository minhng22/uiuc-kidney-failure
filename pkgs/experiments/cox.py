import os
import joblib
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.utils import concordance_index

from pkgs.commons import egfr_tv_cox_model_path, egfr_ti_cox_model_path, hg_cox_model_path, egfr_components_cox_model_path, fivelabms_cox_model_path, heterogen_impute_cox_model_path, ckd_fifty_features_heterogeneous_cox_model_path, four_features_cox_model_path, eight_features_cox_model_path, twenty_features_heterogeneous_cox_model_path, ckd_fifty_features_heterogeneous_train_data_path
from pkgs.data_analysis.model_data_store import get_train_test_data
from pkgs.data_analysis.types import ExperimentScenario
from pkgs.data_analysis.auc_evaluation import report_auc
from pkgs.experiments.utils import round_metric, load_pkl_and_dill_model, compute_brier_score_from_risk_scores, get_tv_rnn_model_features
import dill


# Columns that are structure/outcome, never covariates. CoxTimeVaryingFitter
# consumes subject_id/start/stop/has_esrd via its own id_col/start_col/
# stop_col/event_col arguments and treats EVERY OTHER column in the frame as
# a covariate -- so duration_in_days (the outcome time the C-Index is then
# scored against) and the unnamed CSV row index that get_train_test_data()'s
# `to_csv(path)` writes both silently entered the design matrix. Confirmed on
# rep1's fitted artifacts: duration_in_days coef +5.115e-05 (p=2.0e-11) for
# four_features, +1.260e-04 for eight_features, +1.578e-05 for
# twenty_features_heterogeneous -- all positive, so predicted hazard rose with
# duration and the C-Index scored that against duration itself. Every other
# model already selects features explicitly via get_tv_rnn_model_features(),
# which is why this was Cox-specific.
COX_NON_COVARIATE_COLS = frozenset({'subject_id', 'duration_in_days', 'start', 'stop', 'has_esrd'})

def get_cox_covariates(scenario: ExperimentScenario, df):
    """Explicit covariate list for a CoxTimeVaryingFitter fit on `scenario`.

    Uses the same per-scenario feature table the other models use. Falls back
    to column exclusion for HETEROGENEOUS_IMPUTE, the one scenario
    run_cox_model() accepts that get_tv_rnn_model_features() has no entry for
    (its frame is built in model_data_store.get_train_test_data() from the
    FIVELABMS subsets with the *_missing columns dropped after imputation, so
    it has no fixed column list to name). Mirrors the exclusion rule already
    used in data_analysis/feature_importance_analysis.py."""
    features = get_tv_rnn_model_features(scenario)
    if features is None:
        features = [c for c in df.columns
                    if c not in COX_NON_COVARIATE_COLS and not c.startswith('Unnamed')]
    missing = [c for c in features if c not in df.columns]
    assert not missing, f'{scenario}: covariates missing from data: {missing}'
    return features

def run_cox_model(scenario: ExperimentScenario):
    assert scenario in [ExperimentScenario.TIME_VARIANT, ExperimentScenario.HETEROGENEOUS, ExperimentScenario.EGFR_COMPONENTS, ExperimentScenario.FIVELABMS, ExperimentScenario.HETEROGENEOUS_IMPUTE, ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS, ExperimentScenario.FOUR_FEATURES, ExperimentScenario.EIGHT_FEATURES, ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS]

    data_train, data_test = get_train_test_data(scenario)

    model_path = get_model_path(scenario)

    trained_model = load_pkl_and_dill_model(model_path)

    if not trained_model:
        model = CoxTimeVaryingFitter(penalizer=1.0)

        covariates = get_cox_covariates(scenario, data_train)
        fit_cols = ['subject_id', 'start', 'stop', 'has_esrd'] + covariates
        print(f'Fitting model on {len(covariates)} covariates: {covariates}\n')
        model.fit(data_train[fit_cols], event_col='has_esrd', id_col='subject_id')

        with open(model_path, 'wb') as f:
            dill.dump(model, f, protocol=4)
    else:
        model = trained_model

    print('Evaluate on test data')

    risk_scores_test = model.predict_partial_hazard(data_test)
    c_index_test = round_metric(concordance_index(data_test['duration_in_days'], -risk_scores_test, data_test['has_esrd']))
    print(f'Concordance Index Test: {c_index_test}')

    # Compute Brier Score
    brier_score = compute_brier_score_from_risk_scores(data_train, data_test, -risk_scores_test.values.flatten())
    if brier_score is not None:
        print(f'Integrated Brier Score Test: {brier_score}')

    report_auc(data_train, data_test, risk_scores_test)

def get_model_path(scenario: ExperimentScenario):
    assert scenario in [ExperimentScenario.NON_TIME_VARIANT, ExperimentScenario.TIME_VARIANT,
                        ExperimentScenario.HETEROGENEOUS, ExperimentScenario.EGFR_COMPONENTS, ExperimentScenario.FIVELABMS, ExperimentScenario.HETEROGENEOUS_IMPUTE, ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS, ExperimentScenario.FOUR_FEATURES, ExperimentScenario.EIGHT_FEATURES, ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS]

    model_path = {
        ExperimentScenario.NON_TIME_VARIANT: egfr_ti_cox_model_path,
        ExperimentScenario.TIME_VARIANT: egfr_tv_cox_model_path,
        ExperimentScenario.HETEROGENEOUS: hg_cox_model_path,
        ExperimentScenario.EGFR_COMPONENTS: egfr_components_cox_model_path,
        ExperimentScenario.FIVELABMS: fivelabms_cox_model_path,
        ExperimentScenario.HETEROGENEOUS_IMPUTE: heterogen_impute_cox_model_path,
        ExperimentScenario.CKD_FIFTY_FEATURES_HETEROGENEOUS: ckd_fifty_features_heterogeneous_cox_model_path,
        ExperimentScenario.FOUR_FEATURES: four_features_cox_model_path,
        ExperimentScenario.EIGHT_FEATURES: eight_features_cox_model_path,
        ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS: twenty_features_heterogeneous_cox_model_path,
    }

    return model_path[scenario]

def run_ti_cox_model():
    data_train, data_test = get_train_test_data(ExperimentScenario.NON_TIME_VARIANT)

    model_path = egfr_ti_cox_model_path

    trained_model = load_pkl_and_dill_model(model_path)

    if not trained_model:
        model = CoxPHFitter()

        print(f'Fitting model:\n')
        model.fit(data_train, duration_col='duration_in_days', event_col='has_esrd')

        with open(model_path, 'wb') as f:
            dill.dump(model, f, protocol=4)
    else:
        model = trained_model

    print('Evaluate on test data')
    risk_scores_test = model.predict_partial_hazard(data_test)
    c_index_test = round_metric(concordance_index(data_test['duration_in_days'], -risk_scores_test, data_test['has_esrd']))
    print(f'Concordance Index Test: {c_index_test}')

    # Compute Brier Score
    brier_score = compute_brier_score_from_risk_scores(data_train, data_test, -risk_scores_test.values.flatten())
    if brier_score is not None:
        print(f'Integrated Brier Score Test: {brier_score}')

    report_auc(data_train, data_test, risk_scores_test)

def run_all():
    print("\nRunning non-time-variant Cox model evaluation...")
    run_ti_cox_model()

    print("\nRunning time-variant Cox model evaluation with time-dependent AUC...")
    run_cox_model(ExperimentScenario.TIME_VARIANT)

    print("\nRunning heterogeneous Cox model evaluation with time-dependent AUC...")
    run_cox_model(ExperimentScenario.HETEROGENEOUS)

    print("\nRunning egfr raw Cox model evaluation with time-dependent AUC...")
    run_cox_model(ExperimentScenario.EGFR_COMPONENTS)

    print("\nRunning fivelabms Cox model evaluation with time-dependent AUC...")
    run_cox_model(ExperimentScenario.FIVELABMS)
    print("\nRunning heterogeneous_impute Cox model evaluation with time-dependent AUC...")
    run_cox_model(ExperimentScenario.HETEROGENEOUS_IMPUTE)

def joblib_to_dill():
    for scenario in [ExperimentScenario.NON_TIME_VARIANT, ExperimentScenario.TIME_VARIANT, ExperimentScenario.HETEROGENEOUS, ExperimentScenario.EGFR_COMPONENTS]:
        model_path = get_model_path(scenario)
        if os.path.exists(model_path):
            # if file path ends with .dill, skip
            if model_path.endswith('.dill'):
                continue
            model = joblib.load(model_path)
            with open(model_path, 'wb') as f:
                dill.dump(model, f, protocol=4)

if __name__ == "__main__":
    print("\nRunning FOUR_FEATURES Cox model evaluation with time-dependent AUC...")
    run_cox_model(ExperimentScenario.FOUR_FEATURES)
    print("\nRunning EIGHT_FEATURES Cox model evaluation with time-dependent AUC...")
    run_cox_model(ExperimentScenario.EIGHT_FEATURES)
    print("\nRunning TWENTY_FEATURES_HETEROGENEOUS Cox model evaluation with time-dependent AUC...")
    run_cox_model(ExperimentScenario.TWENTY_FEATURES_HETEROGENEOUS)
