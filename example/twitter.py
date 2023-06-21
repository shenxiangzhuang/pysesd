import pandas as pd

from pysesd.sesd import SESD


def get_ts():
    """
    The get_ts function creates a random time series with two outliers.

    :return: A time series
    """
    ts = pd.read_csv(
        "../dataset/twitter_raw_data.csv",
        parse_dates=["timestamp"],
        index_col=["timestamp"],
    )
    ts = ts.iloc[:, 0]
    ts.index = pd.DatetimeIndex(ts.index)
    return ts


def run():
    """
    The run function is the main function of this module.
    It will be called by the user to run the SESD algorithm on a time series.

    :return: The list of outliers
    """
    ts = get_ts()
    sesd = SESD(alpha=0.05, hybrid=True, max_outliers=int(len(ts) * 0.02))
    outliers = sesd.fit(ts)
    print(f"Got {len(outliers)} in {len(ts)} points, anomaly index: {outliers}")
    sesd.plot(save=True, fig_dir="../figures", fig_name="twitter.png")


if __name__ == "__main__":
    run()
