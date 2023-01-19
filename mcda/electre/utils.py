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

    :param a_value: alternative's performance value on one criterion
    :param b_value: alternative's performance value on the same criterion as `a_value`
    :param scale: criterion's scale with specified preference direction

    :return: difference between criterion values;
        if positive, that means the `a` alternative is better than the `b`,
        ``0`` means there's no difference in performances, and negative value
        implies the `b` is better than `a`.
    """
    _both_values_in_scale(a_value, b_value, scale)
    return (
        a_value - b_value
        if scale.preference_direction == PreferenceDirection.MAX
        else b_value - a_value
    )


def is_veto_exceeded(
    a_values: pd.Series,
    b_values: pd.Series,
    scales: pd.Series,
    veto_thresholds: pd.Series,
) -> bool:
    """Determines if veto threshold is exceeded when comparing
    any pair of alternatives (or alternative and profile).

    :param a_values: alternative's performance on all its criteria
    :param b_values: alternative's performance on all its criteria
    :param scales: all criteria's scales with specified preference direction
    :param veto_thresholds: all criteria's veto thresholds

    :return: ``True`` if is veto was exceeded between a and b, ``False`` otherwise
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
    """Calculates linear function value.

    .. math::
        y = \\alpha \\cdot x + \\beta

    :param alpha: coefficient of the independent variable
    :param x: independent variable
    :param beta: y-intercept

    :return: Dependent variable
    """
    try:
        return alpha * x + beta
    except TypeError as exc:
        if isinstance(x, int) or isinstance(x, float):
            exc.args = (
                "Wrong alpha or beta coefficient. Expected numeric types, but got "
                f"alpha: {type(alpha).__name__}, beta: {type(beta).__name__} instead.",
            )
        else:
            exc.args = (
                "Wrong linear function argument type. Expected numeric, but got "
                f"{type(x).__name__} instead.",
            )
        raise


def transform_series(series: pd.Series) -> pd.Series:
    """Flattens pandas Series and swaps keys with values.

    :param series: pandas Series object

    :return: Transformed pandas Series
    """
    try:
        series = series.explode()
        return pd.Series(series.index.values, index=series)
    except AttributeError as exc:
        raise TypeError(
            f"Wrong argument type. Expected {pd.Series.__name__}, "
            f"but got {type(series).__name__} instead."
        ) from exc


def order_to_outranking_matrix(order: pd.Series) -> pd.DataFrame:
    """Transforms order (upward or downward) to outranking matrix.

    :param order: nested list with order (upward or downward)

    :return: Outranking matrix of given order
    """
    try:
        if set(order.keys()) != {x for x in range(1, len(order) + 1)}:
            raise exceptions.InconsistentIndexNamesError(
                "Values in the upward or downward order should be "
                "a sequential integers, starting with 1, but got "
                f"{set(order.keys())} instead."
            )
    except AttributeError as exc:
        raise TypeError(
            f"Wrong order type. Expected {pd.Series.__name__}, but "
            f"got {type(order).__name__} instead."
        ) from exc
    alternatives = order.explode().to_list()
    if len(set(alternatives)) != len(alternatives):
        raise exceptions.InconsistentIndexNamesError(
            "In an upward or downward order, one alternative cannot "
            "belong to more than one lists."
        )

    outranking_matrix = pd.DataFrame(0, index=alternatives, columns=alternatives)

    for position in order:
        outranking_matrix.loc[position, position] = 1
        outranking_matrix.loc[position, alternatives[alternatives.index(position[-1]) + 1:]] = 1

    return outranking_matrix


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
