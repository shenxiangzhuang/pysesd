import numpy as np
import pandas as pd


def load_synthetic_ts() -> pd.Series:
    """
    The get_ts function creates a random time series with two outliers.

    :return: A time series
    """
    dates = pd.date_range(start="2023-01-01", end="2023-06-19", freq="D")
    values = np.random.random(len(dates))
    values[42] = 10
    values[24] = 20

    ts = pd.Series(data=values, index=dates)
    return ts


def load_twitter_ts():
    """
    The get_ts function creates a random time series with two outliers.

    :return: A time series
    """
    ts = pd.read_csv(
        "../dataset/twitter_raw_data.csv",
        parse_dates=True,
        index_col=["timestamp"],
    )
    ts = ts.iloc[:, 0]
    ts.index = pd.DatetimeIndex(ts.index)
    return ts
