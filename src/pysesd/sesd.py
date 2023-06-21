import logging
import pathlib
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pysesd.util import calculate_stl_residual, esd_test

logger = logging.getLogger(__name__)


class SESD:
    def __init__(
        self,
        alpha: float = 0.05,
        hybrid: bool = False,
        period: Optional[int] = None,
        max_outliers: Optional[int] = None,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the attributes of an instance of a class.

        :param alpha: Set the significance level
        :param hybrid: Determine whether the algorithm is hybrid or not
        :param period: Specify the period of the time series
        :param max_outliers: Specify the maximum number of outliers that can be detected
        :param : Set the alpha value for the test
        :return: None
        """
        self.alpha = alpha
        self.hybrid = hybrid
        self.period = period
        self.max_outliers = max_outliers
        # for plot
        self.outliers: Optional[List[int]] = None
        self.ts: Optional[pd.Series] = None

    def fit(self, ts: pd.Series) -> List[int]:
        # TODO: ts index/NA check
        """
        The fit function takes a time series and returns the indices of outliers.

        :param ts: pd.Series: Pass in the time series data
        :return: The list of outliers
        """
        # data check: ts
        if not isinstance(ts.index, pd.DatetimeIndex):
            try:
                ts.index = pd.DatetimeIndex(ts.index)
            except Exception as e:
                msg = f"The ts data's type is {type(ts.index)} which can't be converted to `pd.DatetimeIndex`: {e}"
                logger.error(msg)
                raise ValueError(msg)

        self.ts = ts
        # parameter check: max_outliers
        n = len(ts)
        if self.max_outliers is None:
            self.max_outliers = 1
        elif self.max_outliers >= n // 2:
            raise ValueError(
                f"max_outliers = {self.max_outliers} >= {n // 2}({n} // 2)"
            )
        # parameter check: period
        if self.period is None:
            self.period = self.__infer_period(ts.index)
        # calc STL residual
        residuals = calculate_stl_residual(np.array(ts.values), period=self.period)
        self.outliers = esd_test(residuals, self.alpha, self.hybrid, self.max_outliers)
        return self.outliers

    @staticmethod
    def __infer_period(datetime_index: pd.DatetimeIndex) -> int:
        """
        Reference: https://github.com/twitter/AnomalyDetection/blob/1f5deaa1609f8f1964c1e905c7a8ad2d1d0dc718/R/ts_anom_detection.R#LL168C1-L173C27  # NOQA
        :param datetime_index: The time series' index
        :return: The inferred period
        """
        infer_freq = pd.infer_freq(datetime_index)
        if infer_freq == "S":  # second -> one day
            period = 86400
        elif infer_freq == "T" or infer_freq == "min":
            period = int(86400 / 60)  # minute -> one day
        elif infer_freq == "H":
            period = int(86400 / 3600)  # hour -> one day
        elif infer_freq == "D":  # day -> one week
            period = 7
        else:  # TODO: more elegant handle method
            period = int(0.1 * len(datetime_index))
        return period

    def plot(
        self, save: bool = True, fig_name: str = "sesd.png", fig_dir: str = "../figures"
    ):
        """
        The plot function plots the time series and the anomaly points.

        :param self: Represent the instance of the class
        :param save: Determine whether the plot should be saved or not
        :param fig_name: Name the file that is saved
        :param fig_dir: Specify the directory where the figure will be saved
        """
        if self.ts is None or self.outliers is None:
            raise ValueError("Please fit the model before plot")
        timestamps = self.ts.index
        values = self.ts.values

        plt.style.use("ggplot")
        _, ax = plt.subplots(figsize=(10, 6))

        # Create a plot for the time series
        ax.plot(timestamps, values, color="lightcoral")

        # Plot the anomaly points
        for point_index in self.outliers:
            ax.plot(
                timestamps[point_index],
                values[point_index],
                color="cyan",
                marker="*",
                markersize=10,
            )

        # Set the title and axis labels
        ax.set_title("S-ESH")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        # Add a legend
        ax.legend(["Time Series", "Anomaly Points"])
        plt.xticks(rotation=60)
        plt.tight_layout()
        if save:
            pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)
            fig_path = f"{fig_dir}/{fig_name}"
            plt.savefig(fig_path, dpi=300)
        plt.show()
