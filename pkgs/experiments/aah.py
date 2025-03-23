import joblib
from lifelines import AalenAdditiveFitter
import os

from pkgs.commons import egfr_tv_aah_model_path
from pkgs.data.model_data_store import get_train_test_data_egfr
from pkgs.experiments.utils import evaluate_scikit_survival_model

# Aalen model assumes time-varying coefficients
def run_tv():
    df, df_test = get_train_test_data_egfr(True)
    X = df[['duration_in_days', 'egfr', 'has_esrd']]

    if not os.path.exists(egfr_tv_aah_model_path):
        model = AalenAdditiveFitter()
        print('Fitting model:')
        model.fit(X, event_col='has_esrd', duration_col='duration_in_days')
        joblib.dump(model, egfr_tv_aah_model_path)
    else:
        print("Loading model from file")
        model = joblib.load(egfr_tv_aah_model_path)

    print('Evaluate on test data')

    # TODO: fix surv_funcs
    X_test = df_test[['duration_in_days', 'egfr', 'has_esrd']]
    surv_funcs_df = model.predict_survival_function(X_test)
    surv_funcs = [surv_funcs_df[col] for col in surv_funcs_df.columns]
    evaluate_scikit_survival_model(df_test, model.predict_cumulative_hazard(df_test).iloc[-1, :], surv_funcs, df)

if __name__ == '__main__':
    run_tv()