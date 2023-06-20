from typing import Optional, List

import matplotlib.pyplot as plt
import pandas as pd

from pysesd.util import esd_test, calculate_stl_residual


class SESD:
    def __init__(self,
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
        self.outliers = None
        self.ts = None

    def fit(self, ts: pd.Series) -> List[int]:
        # TODO: ts index/NA check
        """
        The fit function takes a time series and returns the indices of outliers.

        :param ts: pd.Series: Pass in the time series data
        :return: The list of outliers
        """
        self.ts = ts
        # parameter check: max_outliers
        n = len(ts)
        if self.max_outliers >= n // 2:
            raise ValueError(f"max_outliers = {self.max_outliers} >= {n // 2}({n} // 2)")
        # parameter check: period
        if self.period is None:
            # self.period = int(1 / calculate_density_highest_frequency(ts.values))
            self.period = int(n * 0.2)
        # calc STL residual
        residuals = calculate_stl_residual(ts.values, period=self.period)
        self.outliers = esd_test(residuals, self.alpha, self.hybrid, self.max_outliers)
        return self.outliers

    def plot(self):
        """
        The plot function plots the time series and any detected anomalies.
        """
        timestamps = self.ts.index
        values = self.ts.values
        _, ax = plt.subplots(figsize=(10, 6))

        # Create a plot for the time series
        ax.plot(timestamps, values)

        # Plot the anomaly points
        for point_index in self.outliers:
            ax.plot(timestamps[point_index], values[point_index], 'r*', markersize=10)

        # Set the title and axis labels
        ax.set_title('S-ESH')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        # Add a legend
        ax.legend(['Time Series', 'Anomaly Points'])
        # Show the plot
        plt.tight_layout()
        plt.savefig("sesd.png", dpi=300)
        plt.show()
