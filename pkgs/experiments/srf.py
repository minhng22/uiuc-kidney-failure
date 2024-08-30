from pkgs.commons import srf_model_path
from pkgs.data.store_model_data import get_train_test_data, mini
from pkgs.experiments.utils import get_y
from pkgs.experiments.validation import eval_duration

import joblib
import numpy as np
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest

import datetime
import os

# Data needs to be time-invariant setup
def run_survival_rf():
    df, df_test = get_train_test_data()
    df['has_esrd'] = df['has_esrd'].astype(bool)
    df = mini(df)
    X = df[['duration_in_days', 'egfr']]
    y = get_y(df)

    if os.path.exists(srf_model_path):
        rsf = joblib.load(srf_model_path)
    else:
        print(f'Fitting Random Survival Forest model. Current time {datetime.datetime.now()}:\n')
        rsf = RandomSurvivalForest(n_jobs= 10, verbose=2)
        rsf.fit(X, y)
        joblib.dump(rsf, srf_model_path)

    # C-index is the most popular metric in the last 14 years by a wide margin for evaluating survival models.
    # ref: https://journal.r-project.org/articles/RJ-2023-009/RJ-2023-009.pdf
    c_index = concordance_index(df['duration_in_days'], -rsf.predict(X), df['has_esrd'])
    print(f'Concordance Index: {c_index}')

    df_test['has_esrd'] = df_test['has_esrd'].astype(bool)
    X_test = df_test[['duration_in_days', 'egfr']]

    c_index_test = concordance_index(df_test['duration_in_days'], -rsf.predict(X_test), df_test['has_esrd'])
    print(f'Concordance Index Test: {c_index_test}')

    eval_duration(df_test)
    eval_duration(df)

    surv_fns = rsf.predict_survival_function(X_test)
    # Initialize a list to store median survival times
    median_survival_times = []

    # Loop through each survival function
    for surv_fn in surv_fns:
        # Convert the survival function to an array of times and probabilities
        times = np.array([time for time in surv_fn.x])
        survival_probs = np.array([prob for prob in surv_fn.y])

        # Find the time where the survival probability drops to 0.5 or below
        median_time = times[survival_probs <= 0.5][0] if any(survival_probs <= 0.5) else np.nan

        # Append the median survival time to the list
        median_survival_times.append(median_time)

    # Convert to numpy array or other format if needed
    median_survival_times = np.array(median_survival_times)

    # Now you have the median survival times for each subject
    print(median_survival_times.shape)