from pkgs.commons import egfr_tv_gbsa_model_path
from pkgs.data.model_data_store import get_train_test_data_egfr
from pkgs.experiments.utils import get_y_for_sckit_survival_model, evaluate_scikit_survival_model
import joblib
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import os


def run_gbsa():
    df, df_test = get_train_test_data_egfr(True)
    
    X = df[['duration_in_days', 'egfr']].to_numpy()
    y = get_y_for_sckit_survival_model(df)
    
    print(df.head())
    
    if os.path.exists(egfr_tv_gbsa_model_path):
        gbsa = joblib.load(egfr_tv_gbsa_model_path)
    else:
        print('Fitting Gradient Boosting Survival Analysis model')
        gbsa = GradientBoostingSurvivalAnalysis(verbose=2)
        gbsa.fit(X, y)
        joblib.dump(gbsa, egfr_tv_gbsa_model_path)
    
    print('Evaluate on test data')
    
    X_test = df_test[['duration_in_days', 'egfr']].to_numpy()
    print(X_test.shape)

    evaluate_scikit_survival_model(df_test, -gbsa.predict(X_test), gbsa.predict_survival_function(X_test), df)

if __name__ == '__main__':
    run_gbsa()

