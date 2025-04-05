import joblib
from lifelines import AalenAdditiveFitter
import os

from pkgs.commons import egfr_tv_aah_model_path
from pkgs.data.model_data_store import get_train_test_data
from pkgs.data.types import ExperimentScenario

# Aalen model assumes time-varying coefficients
def run_tv():
    df, df_test = get_train_test_data(ExperimentScenario.TIME_VARIANT)
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

if __name__ == '__main__':
    run_tv()