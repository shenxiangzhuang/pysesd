from pysesd.dataset import load_twitter_ts
from pysesd.sesd import SESD


def run():
    """
    The run function is the main function of this module.
    It will be called by the user to run the SESD algorithm on a time series.

    :return: The list of outliers
    """
    ts = load_twitter_ts()
    sesd = SESD(alpha=0.05, hybrid=True, max_outliers=int(len(ts) * 0.02))
    outliers = sesd.fit(ts)
    sesd.plot(save=True, fig_dir="../figures", fig_name="twitter.png")
    print(f"Got {len(outliers)} in {len(ts)} points, anomaly index: {outliers}")


if __name__ == "__main__":
    run()
