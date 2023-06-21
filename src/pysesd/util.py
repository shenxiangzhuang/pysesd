from typing import List, Optional

import numpy as np
import scipy
from statsmodels.tsa.seasonal import STL


def esd_test(
    data: np.array,
    alpha: float = 0.05,
    hybrid: bool = False,
    max_outliers: Optional[int] = None,
) -> List[int]:
    """
    Perform the Extreme Studentized Deviate (ESD) test to detect potential outliers in the data.

    :param data: The input data to test for potential outliers.
    :param alpha: The significance level for the test. Default is 0.05.
    :param hybrid: Use hybrid method or not. Default is False.
    :param max_outliers: The maximum number of outliers to search for. Default is None,
    which sets max_outliers to len(data) // 2.
    :return: The indices of the detected outliers in the input data.
    """
    if max_outliers is None:
        max_outliers = len(data) // 2

    data = np.ma.array(data)
    outliers = []

    for _ in range(max_outliers):
        test_statistic, test_statistic_idx = calc_test_statistic(data, hybrid)

        # Compute the critical value
        n = len(data) - data.mask.sum()
        t_value = scipy.stats.t.ppf(1 - alpha / (2 * n), n - 2)
        critical_value = (
            (n - 1)
            * t_value
            / np.sqrt(np.square((n - 2)) + (n - 2) * np.square(t_value))
        )

        # Compare the test statistic with the critical value
        if test_statistic > critical_value:
            outliers.append(test_statistic_idx)
            data[test_statistic_idx] = np.ma.masked
        else:
            break

    return outliers


def calc_test_statistic(data: np.ma.array, hybrid: bool = False) -> (float, int):
    """
    The calc_test_statistic function calculates the test statistic for a given data set.

    :param data: Pass the data to be tested
    :param hybrid: Determine whether to use the mean or median(hybrid) for the location parameter
    :return: A tuple of the test statistic and its index
    """
    if hybrid:
        loc_value = np.mean(data)
        scale_value = np.std(data)
    else:
        loc_value = np.median(data)
        scale_value = np.median(np.abs(data - loc_value))

    abs_dev_value = np.abs(data - loc_value)
    max_dev_value_index = np.argmax(abs_dev_value)
    max_dev_value = abs_dev_value[max_dev_value_index]
    test_statistic = max_dev_value / scale_value
    test_statistic_index = max_dev_value_index

    return test_statistic, test_statistic_index


def calculate_stl_residual(
    data: np.ndarray, period: Optional[int] = None
) -> np.ndarray:
    """
    The calculate_stl_residual function takes a time series and returns the STL residuals.

    :param data: Pass the data to be used in the stl decomposition
    :param period: Specify the period of the time series
    :return: The residuals of the stl decomposition
    """
    stl = STL(data, period=period, robust=True)
    decomposition = stl.fit()
    residual = data - decomposition.seasonal - np.median(data)
    return residual
