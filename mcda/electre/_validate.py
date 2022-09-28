from typing import Collection, List, Tuple, get_args

from ..core.aliases import NumericValue
from ..core.scales import PreferenceDirection, QuantitativeScale


def _both_values_in_scale(
    aval: NumericValue, bval: NumericValue, scale: QuantitativeScale
) -> None:
    """Checks if both values are inside the given scale interval.

    :raises ValueError:
        * if `aval` or `bval` is outside its scale interval

    :raises TypeError:
        * if `aval` or `bval` is not a numeric value
        * if `scale` is not a `QuantitativeScale` object
    """
    if not isinstance(scale, QuantitativeScale):
        raise TypeError(
            "Wrong scale type. Expected "
            f"'{_both_values_in_scale.__annotations__['scale'].__name__}', "
            f"but got '{type(scale).__name__}' instead.",
        )

    try:
        if aval not in scale or bval not in scale:
            raise ValueError(
                "Both criteria values must be between the "
                "min and max of the given interval."
            )
    except TypeError as exc:
        exc.args = (
            "Both criteria values must be numeric values, but "
            f"got {type(aval).__name__, type(bval).__name__} instead.",
        )
        raise


def _all_lens_equal(**kwargs: Collection) -> None:
    """Checks if all lists given in args have the same length
    as the base one.

    :raises ValueError:
        * if any list from args has a different length than the first
          kwargs element value

    :raises TypeError:
        * if any argument has no ``len`` function
    """
    args_names, args_values = [list(kwargs.keys()), list(kwargs.values())]
    for name, value in zip(args_names, args_values):
        try:
            if len(value) != len(args_values[0]):
                raise ValueError(
                    f"All lists provided in arguments should have the same length, "
                    f"but len({name})={len(value)}, len({args_names[0]})={len(args_values[0])}."
                )
        except TypeError as exc:
            exc.args = (
                f"Wrong '{name}' argument type. Expected 'Collection', but got "
                f"'{type(value).__name__}' instead.",
            )
            raise


def _inverse_values(
    aval: NumericValue, bval: NumericValue, scale: QuantitativeScale, inverse: bool
) -> Tuple[NumericValue, NumericValue, QuantitativeScale]:
    """Inverses values and the preference direction, if inverse
    parameter is set to True.

    :raises TypeError:
        * if preference direction of the given scale was not provided

    :return Tuple[NumericValue, NumericValue, QuantitativeScale]:
        * if inverse was set to True - aval and bval are switched places
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
            return bval, aval, scale
        return aval, bval, scale
    except AttributeError as exc:
        raise TypeError(
            f"Wrong scale type. Expected <{_inverse_values.__annotations__['scale'].__name__}>, "
            f"but got <{type(scale).__name__}> instead."
        ) from exc


def _weights_proper_vals(weights: Collection[NumericValue]) -> None:
    """Checks if all weights are >= 0

    :raises ValueError:
        * if any weight is less than 0

    :raises TypeError:
        * if any weight is not a numeric type
    """
    try:
        if not all(weight >= 0 for weight in weights):
            raise ValueError("Weight value cannot be negative.")
    except TypeError as exc:
        non_numeric = [
            weight
            for weight in weights
            if not isinstance(weight, get_args(NumericValue))
        ][0]
        exc.args = (
            "All weights values must be a numeric type, but got "
            f"'{type(non_numeric).__name__}' instead.",
        )
        raise


def _reinforcement_factors_vals(reinforcement_factors: List[NumericValue]) -> None:
    """Checks if all reinforcement factors are >= 1

    :raises ValueError:
        * if any factor is less than 1

    :raises TypeError:
        * if any factor is not a numeric type
    """
    try:
        if not all(factor >= 1 for factor in reinforcement_factors):
            raise ValueError("Reinforcement factor cannot be less than 1.")
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

    :raises ValueError:
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
            raise ValueError(
                f"Wrong {name} value. Expected value from a {interval_str} interval, "
                f"but got {value} instead."
            )
    except TypeError as exc:
        exc.args = (
            f"Wrong {name} type. Expected numeric, but got '{type(value).__name__}' instead.",
        )
        raise
