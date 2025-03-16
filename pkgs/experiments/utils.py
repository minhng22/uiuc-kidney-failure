import numpy as np
from lifelines.utils import concordance_index
from sksurv.metrics import integrated_brier_score


def get_y(df):
    return np.array(list(zip(df['has_esrd'].astype(bool), df['duration_in_days'])),
              dtype=[('event', bool), ('time', np.float64)])


def report_metric(metric_num):
    return round(metric_num, 3)


def evaluate(df_test, risk_scores, surv_funcs, y_train):
    # Concordance Index on test data
    c_index_test = report_metric(concordance_index(df_test['duration_in_days'], risk_scores, df_test['has_esrd']))
    print(f'Concordance Index Test: {report_metric(c_index_test)}')
    
    # Brier score on test data
    times_test = np.linspace(0, df_test['duration_in_days'].max(), 100, endpoint=False)
    pred_surv_test = np.asarray([fn(times_test) for fn in surv_funcs])
    ibs_test = integrated_brier_score(y_train, get_y(df_test), pred_surv_test, times_test)
    print(f'Integrated Brier Score (Test): {report_metric(ibs_test)}')