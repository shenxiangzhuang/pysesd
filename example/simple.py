import pandas as pd
import numpy as np

from pysesd.sesd import SESD


def get_ts():
    """
    The get_ts function creates a random time series with two outliers.

    :return: A time series
    """
    dates = pd.date_range(start='2023-01-01', end='2023-06-19', freq='D')
    values = np.random.random(len(dates))
    values[42] = 10
    values[24] = 20

    ts = pd.Series(data=values, index=dates)
    return ts


def run():
    """
    The run function is the main function of this module.
    It will be called by the user to run the SESD algorithm on a time series.


    :return: The list of outliers
    """
    ts = get_ts()
    sesd = SESD(alpha=0.05, hybrid=False, max_outliers=2)
    outliers = sesd.fit(ts)
    print(f"Got {len(outliers)} in {len(ts)} points, anomaly index: {outliers}")
    sesd.plot(save=True, fig_dir="../figures", fig_name="simple.png")


if __name__ == '__main__':
    run()
