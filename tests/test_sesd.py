import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from pysesd.dataset import load_synthetic_ts
from pysesd.sesd import SESD


def test_synthetic_ts():
    """
    The run function is the main function of this module.
    It will be called by the user to run the SESD algorithm on a time series.

    :return: The list of outliers
    """
    ts = load_synthetic_ts()
    sesd = SESD(alpha=0.05, hybrid=True, max_outliers=int(len(ts) * 0.02))
    outliers = sesd.fit(ts)
    assert isinstance(outliers, list)


@given(st.lists(st.floats(), min_size=100, max_size=1000))
@settings(max_examples=10)
def test_random_ts(values):
    dates = pd.date_range(start="2000-01-01", end="2023-01-01", freq="D")[: len(values)]
    ts = pd.Series(data=values, index=dates)
    sesd = SESD(alpha=0.05, hybrid=True, max_outliers=int(len(ts) * 0.02))
    outliers = sesd.fit(ts)
    assert isinstance(outliers, list)
