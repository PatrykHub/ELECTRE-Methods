from typing import (
    Any,
    Collection,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    get_args,
)

import pandas as pd

from ..core.aliases import NumericValue
from ..core.functions import Threshold
from ..core.scales import PreferenceDirection, QuantitativeScale
from . import exceptions


def _both_values_in_scale(
    a_value: NumericValue, b_value: NumericValue, scale: QuantitativeScale
) -> None:
    """Checks if both values are inside the given scale interval.

    :raises ValueOutsideScaleError (ValueError):
        * if `a_value` or `b_value` is outside its scale interval

    :raises TypeError:
        * if `a_value` or `b_value` is not a numeric value
        * if `scale` is not a `QuantitativeScale` object
    """
    if not isinstance(scale, QuantitativeScale):
        raise TypeError(
            "Wrong scale type. Expected "
            f"'{_both_values_in_scale.__annotations__['scale'].__name__}', "
            f"but got '{type(scale).__name__}' instead.",
        )

    try:
        if a_value not in scale or b_value not in scale:
            raise exceptions.ValueOutsideScaleError(
                "Both criteria values must be between the min and max of the given interval."
            )
    except TypeError as exc:
        exc.args = (
            "Both criteria values must be numeric values, but "
            f"got {type(a_value).__name__, type(b_value).__name__} instead.",
        )
        raise


def _get_threshold_values(value: NumericValue, **kwargs: Threshold) -> List[NumericValue]:
    """Computes all threshold values for given `value` argument.

    :raises TypeError:
        * if any threshold is not callable

    :return: list with calculated threshold values, arrange in the
    same order as thresholds
    """
    result: List[NumericValue] = []
    try:
        for name, threshold in kwargs.items():
            result.append(threshold(value))
    except TypeError as exc:
        exc.args = (
            f"Wrong {name} type. Expected {Threshold.__name__}, but "
            f"got {type(threshold).__name__} instead.",
        )
        raise
    return result


def _unique_names(
    names_set: Collection, names_type: Literal["criteria", "alternatives", "profiles"]
) -> None:
    """Checks if passed `names_set` contains only
    unique values

    :param names_type: type of checked values set
    (just to build clear message for the user)

    :raises NotUniqueNamesError (ValueError):
        * if values inside `names_set` are not unique
    """
    if len(set(names_set)) != len(names_set):
        raise exceptions.NotUniqueNamesError(f"Names of {names_type} must contain unique values.")


def _check_df_index(
    df_to_check: pd.DataFrame, index_type: Literal["criteria", "alternatives", "profiles"]
) -> None:
    """Checks if index in `pd.DataFrame` contains only
    unique values.

    :raises NotUniqueNamesError (ValueError):
        * if index values are not unique

    :raises TypeError:
        * if `df_to_check` is not a `pd.DataFrame` type
        (doesn't have the `.index.values` attribute)
    """
    try:
        _unique_names(df_to_check.index.values, names_type=index_type)
    except AttributeError as exc:
        raise TypeError(
            f"Wrong argument type. Expected {pd.DataFrame.__name__}, "
            f"but got {type(df_to_check).__name__} instead."
        ) from exc


def _consistent_criteria_names(**kwargs: Union[Dict, pd.Series, pd.DataFrame]) -> None:
    """Checks if all dictionaries / series contain the same set of keys.

    :raises InconsistentCriteriaNamesError (ValueError):
        * if criteria names are inconsistent, i.e. contain different values set

    :raises NotUniqueNamesError (ValueError):
        * because inside `pd.Series` object there's a possibility for multiple
        existences of the same `key` value, if something like this occurs, the
        error will be raised as well

    :raises TypeError:
        * if any kwarg has no ``keys`` method
    """
    args = list(kwargs.items())
    try:
        i = 0
        _unique_names(args[i][1].keys(), names_type="criteria")
        base_criteria_set = set(args[i][1].keys())

        for i in range(1, len(args)):
            _unique_names(args[i][1].keys(), names_type="criteria")

            if base_criteria_set != set(args[i][1].keys()):
                raise exceptions.InconsistentCriteriaNamesError(
                    "All arguments should have the same criteria names, but found "
                    f"{base_criteria_set} inside the {args[0][0]} argument and "
                    f"{set(args[i][1].keys())} inside the {args[i][0]} argument."
                )
    except AttributeError as exc:
        raise TypeError(
            f"Wrong {args[i][0]} type. Expected "
            f"{_consistent_criteria_names.__annotations__['kwargs']}, "
            f"but got {type(args[i][1]).__name__} instead."
        ) from exc


def _inverse_values(
    a_value: NumericValue, b_value: NumericValue, scale: QuantitativeScale, inverse: bool
) -> Tuple[NumericValue, NumericValue, QuantitativeScale]:
    """Inverses values and the preference direction, if inverse
    parameter is set to True.

    :raises TypeError:
        * if preference direction of the given scale was not provided

    :return Tuple[NumericValue, NumericValue, QuantitativeScale]:
        * if inverse was set to True - a_value and b_value are switched places
          (+ preference direction is changed)
    """
    try:
        if not hasattr(scale, "preference_direction"):
            raise AttributeError
        if inverse:
            scale.preference_direction = (
                PreferenceDirection.MIN
                if scale.preference_direction == PreferenceDirection.MAX
                else PreferenceDirection.MAX
            )
            return b_value, a_value, scale
        return a_value, b_value, scale
    except AttributeError as exc:
        raise TypeError(
            f"Wrong scale type. Expected <{_inverse_values.__annotations__['scale'].__name__}>, "
            f"but got <{type(scale).__name__}> instead."
        ) from exc


def _weights_proper_vals(weights: Union[Dict[Any, NumericValue], pd.Series]) -> None:
    """Checks if all weights are > 0

    :raises WrongWeightValueError (ValueError):
        * if any weight is not positive

    :raises TypeError:
        * if any weight is not a numeric type
    """
    try:
        if not all(
            weight > 0
            for weight in (weights.values() if isinstance(weights, dict) else weights.values)
        ):
            raise exceptions.WrongWeightValueError("Weight value must be positive.")
    except TypeError as exc:
        non_numeric = [
            weight for weight in weights if not isinstance(weight, get_args(NumericValue))
        ][0]
        exc.args = (
            "All weights values must be a numeric type, but got "
            f"'{type(non_numeric).__name__}' instead.",
        )
        raise


def _reinforcement_factors_vals(
    reinforcement_factors: Union[Dict[Any, Optional[NumericValue]], pd.Series]
) -> None:
    """Checks if all reinforcement factors are > 1

    :raises WrongFactorValueError (ValueError):
        * if any factor is less or equal 1

    :raises TypeError:
        * if any factor is not a numeric type and is not ``None``
    """
    try:
        if not all(
            factor > 1 if factor is not None and pd.notna(factor) else True
            for factor in (
                reinforcement_factors.values()
                if isinstance(reinforcement_factors, dict)
                else reinforcement_factors.values
            )
        ):
            raise exceptions.WrongFactorValueError("Reinforcement factor must be greater than 1.")
    except TypeError as exc:
        non_numeric = [
            factor
            for factor in reinforcement_factors
            if not isinstance(factor, get_args(NumericValue))
        ][0]
        exc.args = (
            "All reinforcement factors must be a numeric type, but got "
            f"'{type(non_numeric).__name__}' instead.",
        )
        raise


def _check_index_value_interval(
    value: NumericValue,
    name: str,
    minimal_val: NumericValue = 0,
    maximal_val: NumericValue = 1,
    include_min: bool = True,
    include_max: bool = True,
) -> None:
    """Checks if provided index value is inside its interval.

    :param value: index value
    :param name: index name to display in the exception message
    (such as concordance, cutting level etc.)
    :param min: minimal index value, defaults to 0
    :param max: maximal index value, defaults to 1
    :param include_min: decides if lower boundary of the interval is
    inside of it, defaults to True
    :param include_max: decides if upper boundary of the interval is
    inside of it, defaults to True

    :raises WrongIndexValueError (ValueError:)
        * if `value` is outside its interval

    :raises TypeError:
        * if `value` is not numeric
    """
    try:
        wrong_value_test_result: bool
        interval_str: str
        if include_min and include_max:
            wrong_value_test_result = value < minimal_val or value > maximal_val
            interval_str = f"[{minimal_val}, {maximal_val}]"
        elif include_min and not include_max:
            wrong_value_test_result = value < minimal_val or value >= maximal_val
            interval_str = f"[{minimal_val}, {maximal_val})"
        elif not include_min and include_max:
            wrong_value_test_result = value <= minimal_val or value > maximal_val
            interval_str = f"({minimal_val}, {maximal_val}]"
        else:
            wrong_value_test_result = value <= minimal_val or value >= maximal_val
            interval_str = f"({minimal_val}, {maximal_val})"

        if wrong_value_test_result:
            raise exceptions.WrongIndexValueError(
                f"Wrong {name} value. Expected value from a {interval_str} interval, "
                f"but got {value} instead."
            )
    except TypeError as exc:
        exc.args = (
            f"Wrong {name} type. Expected numeric, but got '{type(value).__name__}' instead.",
        )
        raise
