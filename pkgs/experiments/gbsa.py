from pkgs.commons import egfr_tv_gbsa_model_path
from pkgs.data.model_data_store import get_train_test_data_egfr
from pkgs.experiments.utils import get_y, report_metric, evaluate
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
    
    # Evaluate using Concordance Index on test data
    print('Evaluate on test data')
    
    X_test = df_test[['duration_in_days', 'egfr']].to_numpy()
    print(X_test.shape)

    evaluate(df_test, -gbsa.predict(X_test), gbsa.predict_survival_function(X_test), y)

if __name__ == '__main__':
    run_gbsa()

