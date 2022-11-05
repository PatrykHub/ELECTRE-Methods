import pandas as pd

from mcda.core.aliases import NumericValue


def linear_function(
    alpha: NumericValue, x: NumericValue, beta: NumericValue
) -> NumericValue:
    """Calculates linear function.

    :param alpha: coefficient of the independent variable
    :param x: independent variable
    :param beta: y-intercept

    :return: Dependent variable
    """
    return alpha * x + beta


def transform_series(series: pd.Series) -> pd.Series:
    """Flattens pandas Series and swaps keys with values.

    :param series: pandas Series object

    :return: Transformed pandas Series
    """
    series = series.explode()
    return pd.Series(series.index.values, index=series)


def reverse_transform_series(series: pd.Series) -> pd.Series:
    """Swaps keys with values and nests swapped values.

    :param series: pandas Series object

    :return: Transformed pandas Series
    """
    reversed_series = pd.Series([], dtype="float64")
    for _, index in series.items():
        ranking_level = []
        for key, value in series.items():
            if index == value:
                ranking_level.append(key)
        reversed_series[index] = ranking_level

    return reversed_series
