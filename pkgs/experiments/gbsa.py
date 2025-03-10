from pkgs.commons import egfr_tv_gbsa_model_path
from pkgs.data.model_data_store import get_train_test_data_egfr, mini
from pkgs.experiments.utils import get_y
import joblib
from lifelines.utils import concordance_index
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from pkgs.experiments.utils import report_metric
import os


def run_gbsa():
    df, df_test = get_train_test_data_egfr(True)
    df = mini(df)
    X = df[['duration_in_days', 'egfr']].to_numpy()
    y = get_y(df)

    print(df.head)
    print(X.shape, y.shape)

    if os.path.exists(egfr_tv_gbsa_model_path):
        gbsa = joblib.load(egfr_tv_gbsa_model_path)
    else:
        print('Fitting Gradient Boosting Survival Analysis model')
        gbsa = GradientBoostingSurvivalAnalysis()
        gbsa.fit(X, y)
        joblib.dump(gbsa, egfr_tv_gbsa_model_path)

    print('Evaluate on training data')
    c_index = report_metric(concordance_index(df['duration_in_days'], -gbsa.predict(X), df['has_esrd']))
    print(f'Concordance Index: {c_index}')

    print('Evaluate on test data')
    df_test['has_esrd'] = df_test['has_esrd'].astype(bool)
    df_test.dropna(inplace=True)
    X_test = df_test[['duration_in_days', 'egfr']].to_numpy()
    print(X_test.shape)

    c_index_test = report_metric(concordance_index(df_test['duration_in_days'], -gbsa.predict(X_test), df_test['has_esrd']))
    print(f'Concordance Index Test: {c_index_test}')


if __name__ == '__main__':
    run_gbsa()