from pkgs.commons import gbsa_model_path
from pkgs.data.store_model_data import get_train_test_data, mini
from pkgs.experiments.utils import get_y
import joblib
import numpy as np
from lifelines.utils import concordance_index
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import os


def run_gbsa():
    df, df_test = get_train_test_data()
    df = mini(df)
    X = df[['duration_in_days', 'egfr']].to_numpy()
    y = get_y(df)

    if os.path.exists(gbsa_model_path):
        gbsa = joblib.load(gbsa_model_path)
    else:
        print('Fitting Gradient Boosting Survival Analysis model')
        gbsa = GradientBoostingSurvivalAnalysis()
        gbsa.fit(X, y)
        joblib.dump(gbsa, gbsa_model_path)

    c_index = concordance_index(df['duration_in_days'], -gbsa.predict(X), df['has_esrd'])
    print(f'Concordance Index: {c_index}')

    df_test['has_esrd'] = df_test['has_esrd'].astype(bool)
    df_test.dropna(inplace=True)
    X_test = df_test[['duration_in_days', 'egfr']].to_numpy()
    print(np.isnan(X_test).any())

    c_index_test = concordance_index(df_test['duration_in_days'], -gbsa.predict(X_test), df_test['has_esrd'])
    print(f'Concordance Index Test: {c_index_test}')


if __name__ == '__main__':
    run_gbsa()