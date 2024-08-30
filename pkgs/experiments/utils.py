import numpy as np


def get_y(df):
    return np.array(list(zip(df['has_esrd'].astype(bool), df['duration_in_days'])),
              dtype=[('event', bool), ('time', np.float64)])


def report_metric(metric_num):
    return round(metric_num, 3)