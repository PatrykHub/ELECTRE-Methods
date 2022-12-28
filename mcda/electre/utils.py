import pandas as pd

from ..core.aliases import NumericValue
from ..core.scales import PreferenceDirection, QualitativeScale
from . import exceptions
from ._validation import _both_values_in_scale, _get_threshold_values


def get_criterion_difference(
    a_value: NumericValue, b_value: NumericValue, scale: QualitativeScale
) -> NumericValue:
    """Calculates criterion difference based on pair criterion values
    including preference direction.

    :param a_value: criterion value of first alternative
    :param b_value: criterion value of second alternative
    :param scale: criterion scale with specified preference direction

    :return: Difference between criterion values
    """
    _both_values_in_scale(a_value, b_value, scale)
    return (
        a_value - b_value
        if scale.preference_direction == PreferenceDirection.MAX
        else b_value - a_value
    )


def is_veto(
    a_values: pd.Series,
    b_values: pd.Series,
    scales: pd.Series,
    veto_thresholds: pd.Series,
) -> bool:
    """Determines if veto is present between two alternatives

    :param a_values: criteria values of first alternative
    :param b_values: criteria values of second alternative
    :param scales: criteria scales with specified preference direction
    :param veto_thresholds: criteria veto thresholds

    :return: ``True`` if is veto between a and b, ``False`` otherwise
    """
    for i in range(len(a_values)):
        if veto_thresholds[i] is not None:
            criterion_difference = get_criterion_difference(a_values[i], b_values[i], scales[i])
            veto_threshold_value = _get_threshold_values(
                a_values[i], veto_threshold=veto_thresholds[i]
            )[0]
            if veto_threshold_value <= 0:
                raise exceptions.WrongThresholdValueError(
                    "Veto threshold value must be positive, but got "
                    f"{veto_threshold_value} instead."
                )

            if criterion_difference > veto_threshold_value:
                return True
    return False


def linear_function(alpha: NumericValue, x: NumericValue, beta: NumericValue) -> NumericValue:
    """Calculates linear function.

    :param alpha: coefficient of the independent variable
    :param x: independent variable
    :param beta: y-intercept

    :return: Dependent variable
    """
    try:
        return alpha * x + beta
    except TypeError as exc:
        exc.args = (
            "Wrong alpha or beta coefficient. Expected numeric types, but got "
            f"alpha: {type(alpha).__name__}, beta: {type(beta).__name__} instead.",
        )
        raise


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
