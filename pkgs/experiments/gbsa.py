from pkgs.commons import egfr_tv_gbsa_model_path
from pkgs.data.model_data_store import get_train_test_data_egfr, mini
from pkgs.experiments.utils import get_y, report_metric
import joblib
from lifelines.utils import concordance_index
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import integrated_brier_score
import numpy as np
import os


def run_gbsa():
    # Load training and test data
    df, df_test = get_train_test_data_egfr(True)
    
    # Prepare training features and outcomes
    X = df[['duration_in_days', 'egfr']].to_numpy()
    y = get_y(df)
    
    print(df.head())
    
    # Load or fit the Gradient Boosting Survival Analysis model
    if os.path.exists(egfr_tv_gbsa_model_path):
        gbsa = joblib.load(egfr_tv_gbsa_model_path)
    else:
        print('Fitting Gradient Boosting Survival Analysis model')
        gbsa = GradientBoostingSurvivalAnalysis()
        gbsa.fit(X, y)
        joblib.dump(gbsa, egfr_tv_gbsa_model_path)
    
    # Evaluate using Concordance Index on training data
    print('Evaluate on training data')
    c_index = report_metric(concordance_index(df['duration_in_days'], -gbsa.predict(X), df['has_esrd']))
    print(f'Concordance Index: {c_index}')
    
    # Evaluate using Concordance Index on test data
    print('Evaluate on test data')
    df_test['has_esrd'] = df_test['has_esrd'].astype(bool)
    df_test.dropna(inplace=True)
    X_test = df_test[['duration_in_days', 'egfr']].to_numpy()
    print(X_test.shape)
    c_index_test = report_metric(concordance_index(df_test['duration_in_days'], -gbsa.predict(X_test), df_test['has_esrd']))
    print(f'Concordance Index Test: {c_index_test}')
    
    # Brier score
    # For test data, define a time grid based on the maximum follow-up time in the test set
    times_test = np.linspace(0, df_test['duration_in_days'].max(), 100)
    
    # Obtain survival functions for test data and evaluate them over times_test
    surv_funcs_test = gbsa.predict_survival_function(X_test)
    pred_surv_test = np.asarray([fn(times_test) for fn in surv_funcs_test])
    
    # Compute IBS on test data (using training data for censoring estimation)
    ibs_test = integrated_brier_score(y, get_y(df_test), pred_surv_test, times_test)
    print(f'Integrated Brier Score (Test): {ibs_test}')

if __name__ == '__main__':
    run_gbsa()

