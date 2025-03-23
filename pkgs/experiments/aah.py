from lifelines.utils import concordance_index
import joblib
from lifelines import AalenAdditiveFitter
import os

from pkgs.commons import egfr_tv_aah_model_path, egfr_ti_aah_model_path
from pkgs.data.model_data_store import get_train_test_data_egfr
from pkgs.experiments.utils import round_metric

# Aalen model assumes time-varying coefficients
def run_tv():
    data_train, data_test = get_train_test_data_egfr(True)

    if not os.path.exists(egfr_tv_aah_model_path):
        model = AalenAdditiveFitter()
        print(f'Fitting model:\n')
        model.fit(data_train, event_col='has_esrd', duration_col='duration_in_days')
        joblib.dump(model, egfr_tv_aah_model_path)
    else:
        model = joblib.load(egfr_tv_aah_model_path)

    print('Evaluate on test data')
    risk_scores_test = model.predict_cumulative_hazard(data_test).iloc[-1, :]
    c_index_test = round_metric(concordance_index(data_test['duration_in_days'], -risk_scores_test, data_test['has_esrd']))
    print(f'Concordance Index Test: {c_index_test}')

if __name__ == '__main__':
    run_tv()