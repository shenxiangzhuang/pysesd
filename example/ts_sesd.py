import pandas as pd
import numpy as np

from pysesd.sesd import SESD


def get_ts():
    # Create some sample data
    dates = pd.date_range(start='2023-01-01', end='2023-06-19', freq='D')
    values = np.random.random(len(dates))
    values[42] = 10
    values[24] = 20

    ts = pd.Series(data=values, index=dates)
    return ts


def run():
    ts = get_ts()
    sesd = SESD(alpha=0.05, hybrid=False, max_outliers=2)
    outliers = sesd.fit(ts)
    print(outliers)
    sesd.plot()


if __name__ == '__main__':
    run()
