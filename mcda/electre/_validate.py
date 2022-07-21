from typing import List, Tuple

from ..core.scales import QuantitativeScale, PreferenceDirection
from ..core.aliases import NumericValue


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
    try:
        if aval not in scale or bval not in scale:
            raise ValueError(
                "Both criteria values must be between the "
                "min and max of the given interval."
            )
    except TypeError as exc:
        if isinstance(scale, QuantitativeScale):
            exc.args = (
                "Both criteria values must be numeric values, but "
                f"got {type(aval).__name__, type(bval).__name__} instead.",
            )
        else:
            exc.args = (
                "Wrong scale type. Expected "
                f"{_both_values_in_scale.__annotations__['scale'].__name__}, "
                f"but got {type(scale).__name__} instead.",
            )
        raise


def _all_lens_equal(base: List, *args) -> None:
    """Checks if all lists given in args have the same length
    as the base one.

    :raises ValueError:
        * if any list from args has a different length than the base

    :raises TypeError:
        * if any argument has no ``len`` function
    """
    try:
        if not all(len(base) == len_other for len_other in [len(arg) for arg in args]):
            raise ValueError("All provided lists need to be the same length.")
    except TypeError as exc:
        exc.args = ("All lists given in arguments have to be lists.",)
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


def _weights_proper_vals(weights: List[NumericValue]) -> None:
    """Checks if all weights are >= 0

    :raises ValueError:
        * if any weight is less than 0

    :raises TypeError:
        * if any weight is not a numeric type
    """
    try:
        if not all(weight >= 0 for weight in weights):
            raise ValueError("Weight value cannot be less than 0.")
    except TypeError as exc:
        exc.args = ("All weights values must be a numeric type.",)
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
        exc.args = ("All factors values must be a numeric type.",)
        raise
