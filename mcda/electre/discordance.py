"""This module implements methods to compute discordance."""
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.aliases import NumericValue
from ..core.functions import Threshold
from ..core.scales import PreferenceDirection, QuantitativeScale
from . import exceptions
from ._validation import (
    _both_values_in_scale,
    _check_df_index,
    _check_index_value_interval,
    _consistent_criteria_names,
    _consistent_df_indexing,
    _get_threshold_values,
    _inverse_values,
    _unique_names,
    _weights_proper_vals,
)


def discordance_bin_marginal(
    a_value: NumericValue,
    b_value: NumericValue,
    scale: QuantitativeScale,
    veto_threshold: Optional[Threshold],
    inverse: bool = False,
) -> int:
    """Computes marginal binary discordance index:
    :math:`d^V(a\\_value, b\\_value) \\in \\{0, 1\\}`,
    which represents an opposition in outranking `a` over `b`
    on the provided criterion.

    :param a_value: alternative's performance value on one criterion
    :param b_value: alternative performance value on the same criterion as `a_value`
    :param scale: criterion's scale with specified preference direction
    :param veto_threshold: criterion's veto threshold
    :param inverse: if ``True``, then function will calculate :math:`d^V(b\\_value, a\\_value)`
        with opposite scale preference direction, defaults to ``False``

    :raises exceptions.WrongThresholdValueError: if the veto threshold
        value is not positive.

    :return: ``0``, if veto was not exceeded or `veto_threshold` is ``None``,
        ``1`` otherwise
    """
    _both_values_in_scale(a_value, b_value, scale)
    a_value, b_value, scale = _inverse_values(a_value, b_value, scale, inverse)

    if veto_threshold is None:
        return 0

    veto_threshold_value = _get_threshold_values(a_value, veto_threshold=veto_threshold)[0]
    if veto_threshold_value <= 0:
        raise exceptions.WrongThresholdValueError(
            f"Veto threshold value must be positive, but got {veto_threshold_value} instead."
        )

    if scale.preference_direction == PreferenceDirection.MAX:
        return 1 if b_value - a_value >= veto_threshold_value else 0
    return 1 if a_value - b_value >= veto_threshold_value else 0


def discordance_bin_comprehensive(
    a_values: Union[Dict[Any, NumericValue], pd.Series],
    b_values: Union[Dict[Any, NumericValue], pd.Series],
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    veto_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
    inverse: bool = False,
    **kwargs,
) -> int:
    """Computes comprehensive binary discordance index:
    :math:`D^V(a, b) \\in \\{0, 1\\}`, which represents an opposition
    in outranking a over b.

    :param a_values: alternative's performance on all its criteria
    :param b_values: alternative's performance on all its criteria
    :param scales: all criteria's scales with specified preference direction
    :param veto_thresholds: all criteria's veto thresholds;
        for each criterion, providing a ``Threshold`` is optional and it
        can be set to ``None``
    :param inverse: if ``True``, then function will calculate :math:`D^V(b, a)`
        with opposite scales preference direction, defaults to ``False``

    :return: ``0``, if veto occur on any criterion, ``1`` otherwise
    """
    if "validated" not in kwargs:
        _consistent_criteria_names(
            a_values=a_values, b_values=b_values, scales=scales, veto_thresholds=veto_thresholds
        )

    for criterion_name in a_values.keys():
        if discordance_bin_marginal(
            a_values[criterion_name],
            b_values[criterion_name],
            scales[criterion_name],
            veto_thresholds[criterion_name],
            inverse,
        ):
            return 1
    return 0


def discordance_bin_criteria_marginals(
    a_values: Union[Dict[Any, NumericValue], pd.Series],
    b_values: Union[Dict[Any, NumericValue], pd.Series],
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    veto_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
    **kwargs,
) -> pd.Series:
    """Computes marginal binary discordance indices:
    :math:`d_j^V(a, b) \\in \\{0, 1\\}` for alternatives pair
    (or between alternative and profile).

    :param a_value: alternative's performance value on one criterion
    :param b_value: alternative's performance value on the same criterion as `a_value`
    :param scale: criterion's scale with specified preference direction
    :param veto_threshold: criterion's veto threshold

    :return: a series with marginal binary discordance for each criterion
    """
    if "validated" not in kwargs:
        _consistent_criteria_names(
            a_values=a_values, b_values=b_values, scales=scales, veto_thresholds=veto_thresholds
        )

    return pd.Series(
        [
            discordance_bin_marginal(
                a_values[criterion_name],
                b_values[criterion_name],
                scales[criterion_name],
                veto_thresholds[criterion_name],
            )
            for criterion_name in a_values.keys()
        ],
        index=list(a_values.keys()),
    )


def discordance_bin(
    alternatives_perform: pd.DataFrame,
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    veto_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
    profiles_perform: Optional[pd.DataFrame] = None,
    return_marginals: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Computes binary discordance indices: marginals or comprehensive ones.
    Comparison is possible for alternatives or between alternatives and
    profiles, depending on the `profiles_perform` argument value.

    :param alternatives_perform: performance table of alternatives
    :param scales: all criteria's scales with specified preference direction
    :param veto_thresholds: all criteria's veto thresholds
    param profiles_perform: if provided, indices would be computed
        between alternatives from `alternatives_perform` and from this argument,
        defaults to ``None``
    :param return_marginals: if ``True``, the function will return
        marginal indices, not the comprehensive ones, defaults to ``False``

    :return:
        * if `profiles_perform` argument is set to ``None``, the function will
          return a single `pandas.DataFrame` object with :math:`D^V(a, b)` indices
          for all alternatives pairs
        * otherwise, the function will return a ``tuple`` object with two
          `pandas.DataFrame` objects inside, where the first one contains
          :math:`D^V(a, b)(alternative_i, profile_j)` indices, and the second one
          :math:`D^V(a, b)(profile_j, alternative_i)`, respectively
        * when `return_marginals` argument is set to ``True``, data frames will
          contain a series with discordance marginals, not the :math:`D^V` indices
    """
    discordance_function: Callable = discordance_bin_comprehensive

    if return_marginals:
        discordance_function = discordance_bin_criteria_marginals

    _consistent_criteria_names(
        alternatives_perform=alternatives_perform,
        profiles_perform=profiles_perform,
        scales=scales,
        veto_thresholds=veto_thresholds,
    )
    _check_df_index(alternatives_perform, index_type="alternatives")
    _check_df_index(profiles_perform, index_type="profiles")
    if profiles_perform is not None:
        return pd.DataFrame(
            [
                [
                    discordance_function(
                        alternatives_perform.loc[alt_name],
                        profiles_perform.loc[prof_name],
                        scales,
                        veto_thresholds,
                        validated=True,
                    )
                    for prof_name in profiles_perform.index.values
                ]
                for alt_name in alternatives_perform.index.values
            ],
            index=alternatives_perform.index,
            columns=profiles_perform.index,
        ), pd.DataFrame(
            [
                [
                    discordance_function(
                        profiles_perform.loc[prof_name],
                        alternatives_perform.loc[alt_name],
                        scales,
                        veto_thresholds,
                        validated=True,
                    )
                    for alt_name in alternatives_perform.index.values
                ]
                for prof_name in profiles_perform.index.values
            ],
            index=profiles_perform.index,
            columns=alternatives_perform.index,
        )

    return pd.DataFrame(
        [
            [
                discordance_function(
                    alternatives_perform.loc[alt_name_a],
                    alternatives_perform.loc[alt_name_b],
                    scales,
                    veto_thresholds,
                    validated=True,
                )
                for alt_name_b in alternatives_perform.index.values
            ]
            for alt_name_a in alternatives_perform.index.values
        ],
        index=alternatives_perform.index,
        columns=alternatives_perform.index,
    )


def discordance_marginal(
    a_value: NumericValue,
    b_value: NumericValue,
    scale: QuantitativeScale,
    preference_threshold: Threshold,
    veto_threshold: Optional[Threshold],
    pre_veto_threshold: Optional[Threshold] = None,
    inverse: bool = False,
) -> NumericValue:
    """Computes marginal discordance index
    :math:`d(a\\_value, b\\_value) \\in [0, 1]`,
    which represents a degree in opposition in outranking `a` over `b`
    on the provided criterion.

    The function calculates
    :math:`d(a\\_value, b\\_value) \\in \\{d^{PV}(a\\_value, b\\_value),
    d^{PVV}(a\\_value, b\\_value)\\}` index, depending
    on the `pre_veto_threshold` presence.

    :param a_value: alternative's performance value on one criterion
    :param b_value: alternative's performance value on the same criterion as `a_value`
    :param scale: criterion's scale with specified preference direction
    :param preference_threshold: criterion's preference threshold
    :param veto_threshold: criterion's veto threshold
    :param pre_veto_threshold: criterion's pre-veto threshold; if provided,
        with ``Threshold``, then the `veto_threshold` must be set as well,
        defaults to ``None``
    :param inverse: if ``True``, then function will calculate :math:`d(b\\_value, a\\_value)`
        with opposite scales preference direction, defaults to ``False``

    :raises exceptions.WrongThresholdValueError: if thresholds don't meet a condition:
    .. math::
        0 \\le preference\\_threshold(value) < veto\\_threshold(value)
        < pre\\_veto\\_threshold(value)

    :return: marginal discordance index, value from the [0, 1] interval;
        if no `veto_threshold` is specified, the function will return ``0``
    """
    _both_values_in_scale(a_value, b_value, scale)
    a_value, b_value, scale = _inverse_values(a_value, b_value, scale, inverse)

    if veto_threshold is None:
        if pre_veto_threshold is not None:
            raise exceptions.WrongThresholdValueError(
                "Missing veto threshold for criterion with defined pre-veto."
            )
        return 0

    preference, veto = _get_threshold_values(
        a_value, preference_threshold=preference_threshold, veto_threshold=veto_threshold
    )
    pre_veto = (
        _get_threshold_values(a_value, pre_veto_threshold=pre_veto_threshold)[0]
        if pre_veto_threshold is not None
        else None
    )

    if preference >= veto:
        raise exceptions.WrongThresholdValueError(
            "Preference threshold must be less than the veto threshold, but got "
            f"preference_threshold={preference}, veto_threshold={veto}."
        )
    if pre_veto and not veto > pre_veto > preference:
        raise exceptions.WrongThresholdValueError(
            "Thresholds must meet a condition: veto > pre_veto > preference, but got "
            f"{veto} > {pre_veto} > {preference} instead."
        )

    discordance_beginning = preference if pre_veto is None else pre_veto
    if scale.preference_direction == PreferenceDirection.MAX:
        if b_value - a_value > veto:
            return 1
        elif b_value - a_value <= discordance_beginning:
            return 0
        return (b_value - a_value - discordance_beginning) / (veto - discordance_beginning)

    if a_value - b_value > veto:
        return 1
    elif a_value - b_value <= discordance_beginning:
        return 0
    return (a_value - b_value - discordance_beginning) / (veto - discordance_beginning)


def discordance_comprehensive(
    a_values: Union[Dict[Any, NumericValue], pd.Series],
    b_values: Union[Dict[Any, NumericValue], pd.Series],
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    weights: Union[Dict[Any, NumericValue], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    veto_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
    pre_veto_thresholds: Optional[Union[Dict[Any, Optional[Threshold]], pd.Series]] = None,
    **kwargs,
) -> NumericValue:
    """Computes comprehensive discordance index :math:`\\Delta^{CD}(a, b) \\in [0, 1]`
    from :math:`d_j^{PV}(a, b)` or :math:`d_j^{PVV}(a, b)` indices,
    depending on a `pre_veto_thresholds` presence for each criterion.

    :param a_values: alternative's performance on all its criteria
    :param b_values: alternative's performance on all its criteria
    :param scales: all criteria's scales with specified preference direction
    :param weights: all criteria's weights, :math:`weight\\_value > 0`
    :param preference_thresholds: all criteria's preference thresholds
    :param veto_thresholds: all criteria's veto thresholds
    :param pre_veto_thresholds: all criteria's pre-veto thresholds,
        defaults to ``None``

    :return: comprehensive discordance index, value from the [0, 1] interval
    """
    if "validated" not in kwargs:
        _consistent_criteria_names(
            a_values=a_values,
            b_values=b_values,
            scales=scales,
            weights=weights,
            preference_thresholds=preference_thresholds,
            veto_thresholds=veto_thresholds,
        )
        _weights_proper_vals(weights)
    return sum(
        pd.Series(weights)
        * discordance_criteria_marginals(
            a_values,
            b_values,
            scales,
            preference_thresholds,
            veto_thresholds,
            pre_veto_thresholds,
            validated=True,
        )
    ) / sum(weights.values() if isinstance(weights, dict) else weights.values)


def discordance_criteria_marginals(
    a_values: Union[Dict[Any, NumericValue], pd.Series],
    b_values: Union[Dict[Any, NumericValue], pd.Series],
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    veto_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
    pre_veto_thresholds: Optional[Union[Dict[Any, Optional[Threshold]], pd.Series]] = None,
    **kwargs,
) -> pd.Series:
    """Computes marginal discordance indices:
    :math:`d_j^{PVV}(a, b) \\in \\{0, 1\\}` or :math:`d_j^{PV}(a, b) \\in \\{0, 1\\}`
    for alternatives pair (or between alternative and profile).

    :param a_values: alternative's performance on all its criteria
    :param b_values: alternative's performance on all its criteria
    :param scales: all criteria's scales with specified preference direction
    :param weights: all criteria's weights, :math:`weight\\_value > 0`
    :param preference_thresholds: all criteria's preference thresholds
    :param veto_thresholds: all criteria's veto thresholds
    :param pre_veto_thresholds: all criteria's pre-veto thresholds,
        defaults to ``None``

    :return: a series with marginal discordance indices
    """
    if "validated" not in kwargs:
        _consistent_criteria_names(
            a_values=a_values,
            b_values=b_values,
            scales=scales,
            preference_thresholds=preference_thresholds,
            veto_thresholds=veto_thresholds,
        )
    return pd.Series(
        [
            discordance_marginal(
                a_values[criterion_name],
                b_values[criterion_name],
                scales[criterion_name],
                preference_thresholds[criterion_name],
                veto_thresholds[criterion_name],
                (pre_veto_thresholds[criterion_name] if pre_veto_thresholds is not None else None),
            )
            for criterion_name in a_values.keys()
        ],
        index=list(a_values.keys()),
    )


def discordance_marginals(
    alternatives_perform: pd.DataFrame,
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    veto_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
    pre_veto_thresholds: Optional[Union[Dict[Any, Optional[Threshold]], pd.Series]] = None,
    profiles_perform: Optional[pd.DataFrame] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Computes marginal discordance indices :math:`d_j^{PV}(a, b)`
    or :math:`d_j^{PVV}(a, b)`. Comparison is possible for
    alternatives or between alternatives and profiles,
    depending on the `profiles_perform` argument value.

    :param alternatives_perform: performance table of alternatives
    :param scales: all criteria's scales with specified preference direction
    :param preference_thresholds: all criteria's preference thresholds
    :param veto_thresholds: all criteria's preference thresholds
    :param pre_veto_thresholds: all criteria's pre-veto thresholds,
        defaults to ``None``
    :param profiles_perform: if provided, indices would be computed
        between alternatives from `alternatives_perform` and from this argument,
        defaults to ``None``

    :return:
        * if `profiles_perform` argument is set to ``None``, the function will
          return a single `pandas.DataFrame` object with `pandas.Series` objects
          inside, each containing marginal discordance indices
        * otherwise, the function will return a ``tuple`` object with two
          `pandas.DataFrame` objects inside, where the first one contains
          :math:`d_j(alternative_i, profile_j)` indices, and the second one
          :math:`d_j(profile_j, alternative_i)`, respectively,
          where :math:`d_j \\in \\{d_j^{PV}, d_j^{PVV}\\}`.
    """
    _consistent_criteria_names(
        alternatives_perform=alternatives_perform,
        scales=scales,
        preference_thresholds=preference_thresholds,
        veto_thresholds=veto_thresholds,
        pre_veto_thresholds=pre_veto_thresholds,
        profiles_perform=profiles_perform,
    )
    _check_df_index(alternatives_perform, index_type="alternatives")
    _check_df_index(profiles_perform, index_type="profiles")
    if profiles_perform is not None:
        return pd.DataFrame(
            [
                [
                    discordance_criteria_marginals(
                        alternatives_perform.loc[alt_name],
                        profiles_perform.loc[prof_name],
                        scales,
                        preference_thresholds,
                        veto_thresholds,
                        pre_veto_thresholds,
                        validated=True,
                    )
                    for prof_name in profiles_perform.index.values
                ]
                for alt_name in alternatives_perform.index.values
            ],
            index=alternatives_perform.index,
            columns=profiles_perform.index,
        ), pd.DataFrame(
            [
                [
                    discordance_criteria_marginals(
                        profiles_perform.loc[prof_name],
                        alternatives_perform.loc[alt_name],
                        scales,
                        preference_thresholds,
                        veto_thresholds,
                        pre_veto_thresholds,
                        validated=True,
                    )
                    for alt_name in alternatives_perform.index.values
                ]
                for prof_name in profiles_perform.index.values
            ],
            index=profiles_perform.index,
            columns=alternatives_perform.index,
        )
    return pd.DataFrame(
        [
            [
                discordance_criteria_marginals(
                    alternatives_perform.loc[alt_name_a],
                    alternatives_perform.loc[alt_name_b],
                    scales,
                    preference_thresholds,
                    veto_thresholds,
                    pre_veto_thresholds,
                    validated=True,
                )
                for alt_name_b in alternatives_perform.index.values
            ]
            for alt_name_a in alternatives_perform.index.values
        ],
        index=alternatives_perform.index,
        columns=alternatives_perform.index,
    )


def discordance(
    discordance_marginals: pd.DataFrame,
    weights: Union[Dict[Any, NumericValue], pd.Series],
) -> pd.DataFrame:
    """Aggregates discordance marginals to comprehensive
    indices :math:`\\Delta^{CD}(a, b) \\in [0, 1]` from
    :math:`d^{V}(a, b)`, :math:`d^{PV}(a, b)` or :math:`d^{PVV}(a, b)`
    indices.

    :param discordance_marginals: a table with marginal indices,
        can be an output from :func:`discordance_marginals` or
        :func:`discordance_bin` functions
    :param weights: all criteria's weights, :math:`weight\\_value > 0`

    :return: a data frame with comprehensive discordance indices
    """
    _check_df_index(discordance_marginals, index_type="rows")
    _check_df_index(discordance_marginals, index_type="columns", check_columns=True)

    try:
        _unique_names(weights.keys(), names_type="criteria")
        _weights_proper_vals(weights)
    except AttributeError as exc:
        raise TypeError(
            f"Wrong weights type. Expected {discordance.__annotations__['weights']}, "
            f"but got {type(weights).__name__} instead."
        ) from exc

    criteria_set = set(weights.keys())
    try:
        for column_name in discordance_marginals.columns.values:
            for row_name in discordance_marginals[column_name].index.values:
                _unique_names(
                    discordance_marginals[column_name][row_name].keys(), names_type="criteria"
                )
                if set(discordance_marginals[column_name][row_name].keys()) != criteria_set:
                    raise exceptions.InconsistentIndexNamesError(
                        "A criteria set inside discordance marginals table is different from "
                        "the set provided with weights."
                    )
                for value in discordance_marginals[column_name][row_name].values:
                    _check_index_value_interval(value, "marginal discordance")

    except AttributeError as exc:
        raise TypeError(
            f"Wrong marginal discordance value type. Expected "
            f"{pd.Series.__name__}, but got "
            f"{type(discordance_marginals[column_name][row_name]).__name__} instead."
        ) from exc

    weights_sum: NumericValue = sum(weights)
    return pd.DataFrame(
        [
            [
                sum(weights * discordance_marginals[alt_name_b][alt_name_a]) / weights_sum
                for alt_name_b in discordance_marginals.columns
            ]
            for alt_name_a in discordance_marginals.index.values
        ],
        index=discordance_marginals.index,
        columns=discordance_marginals.columns,
    )


class NonDiscordanceType(Enum):
    DC = "DC"
    D = "D"
    DM = "DM"


def non_discordance_marginal(
    criteria_discordance_marginals: pd.Series,
    non_discordance_type: NonDiscordanceType = NonDiscordanceType.DC,
    concordance_comprehensive: Optional[NumericValue] = None,
) -> NumericValue:
    try:
        _unique_names(criteria_discordance_marginals.keys(), names_type="criteria")

        for value in criteria_discordance_marginals.values:
            _check_index_value_interval(value, name="marginal discordance")
    except AttributeError as exc:
        raise TypeError(
            f"Wrong marginal discordance indices type. Expected {pd.Series.__name__}, "
            f"but got {type(criteria_discordance_marginals).__name__} instead."
        ) from exc

    if non_discordance_type == NonDiscordanceType.DM:
        return 1 - max(criteria_discordance_marginals)

    if non_discordance_type == NonDiscordanceType.D:
        return np.prod(1 - criteria_discordance_marginals)

    if non_discordance_type == NonDiscordanceType.DC:
        if concordance_comprehensive is None:
            raise exceptions.WrongIndexValueError(
                "Missing comprehensive concordance index while computing "
                "the non-discordance DC index value."
            )
        _check_index_value_interval(concordance_comprehensive, name="comprehensive concordance")

        return np.prod(
            (
                1
                - criteria_discordance_marginals[
                    criteria_discordance_marginals > concordance_comprehensive
                ]
            )
            / (1 - concordance_comprehensive)
        )

    raise TypeError(
        f"Non-discordance type was not provided. Expected {NonDiscordanceType.__name__}, "
        f"but got {type(non_discordance_type).__name__} instead."
    )


def non_discordance(
    discordance_marginals: pd.DataFrame,
    non_discordance_type: NonDiscordanceType = NonDiscordanceType.DC,
    concordance_comprehensive: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:

    _consistent_df_indexing(
        discordance_marginals=discordance_marginals,
        concordance_comprehensive=concordance_comprehensive,
    )

    return pd.DataFrame(
        [
            [
                non_discordance_marginal(
                    discordance_marginals.loc[alt_name_a, alt_name_b],
                    non_discordance_type,
                    concordance_comprehensive.loc[alt_name_a, alt_name_b]
                    if concordance_comprehensive is not None
                    else None,
                )
                for alt_name_b in discordance_marginals.columns.values
            ]
            for alt_name_a in discordance_marginals.index.values
        ],
        index=discordance_marginals.index,
        columns=discordance_marginals.columns,
    )


def is_counter_veto_occur(
    a_value: NumericValue,
    b_value: NumericValue,
    scale: QuantitativeScale,
    counter_veto_threshold: Optional[Threshold],
) -> bool:
    """_summary_

    :param a_value: _description_
    :param b_value: _description_
    :param scale: _description_
    :param counter_veto_threshold: _description_

    :return: _description_
    """
    _both_values_in_scale(a_value, b_value, scale)
    if counter_veto_threshold is None:
        return False

    counter_veto_value = _get_threshold_values(
        b_value, counter_veto_threshold=counter_veto_threshold
    )[0]
    if counter_veto_value <= 0:
        raise exceptions.WrongThresholdValueError(
            "Counter veto threshold value must be positive, but "
            f"got {counter_veto_value} instead."
        )
    return (
        a_value - b_value > counter_veto_value
        if scale.preference_direction == PreferenceDirection.MAX
        else b_value - a_value > counter_veto_value
    )


def counter_veto_pair(
    a_values: Union[Dict[Any, NumericValue], pd.Series],
    b_values: Union[Dict[Any, NumericValue], pd.Series],
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    counter_veto_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
    **kwargs,
) -> pd.Series:
    """_summary_

    :param a_values: _description_
    :param b_values: _description_
    :param scales: _description_
    :param counter_veto_thresholds: _description_

    :return: _description_
    """
    if "validated" not in kwargs:
        _consistent_criteria_names(
            a_values=a_values,
            b_values=b_values,
            scales=scales,
            counter_veto_thresholds=counter_veto_thresholds,
        )

    return pd.Series(
        [
            is_counter_veto_occur(
                a_values[criterion_name],
                b_values[criterion_name],
                scales[criterion_name],
                counter_veto_thresholds[criterion_name],
            )
            for criterion_name in a_values.keys()
        ],
        index=list(a_values.keys()),
    )


def counter_veto(
    performance_table: pd.DataFrame,
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    counter_veto_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
    profiles_perform: Optional[pd.DataFrame] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """_summary_

    :param alt_values: _description_
    :param performance_table: _description_
    :param scales: _description_
    :param counter_veto_thresholds: _description_

    :return: _description_
    """
    _consistent_criteria_names(
        performance_table=performance_table,
        profiles_perform=profiles_perform,
        scales=scales,
        counter_veto_thresholds=counter_veto_thresholds,
    )
    _check_df_index(performance_table, index_type="alternatives")
    _check_df_index(profiles_perform, index_type="profiles")

    if profiles_perform is not None:
        result_alt_profs = pd.DataFrame(
            [[[] * len(performance_table.index)] * len(profiles_perform.index)],
            index=performance_table.index,
            columns=profiles_perform.index,
        )
        for criterion_name_a in performance_table.index.values:
            for criterion_name_b in profiles_perform.index.values:
                cv_series = counter_veto_pair(
                    performance_table.loc[criterion_name_a],
                    profiles_perform.loc[criterion_name_b],
                    scales,
                    counter_veto_thresholds,
                    validated=True,
                )
                result_alt_profs[criterion_name_b][criterion_name_a] = [
                    cv_name for cv_name, cv_value in cv_series.items() if cv_value
                ]

        result_profs_alt = pd.DataFrame(
            [[[] * len(profiles_perform.index)] * len(performance_table.index)],
            index=profiles_perform.index,
            columns=performance_table.index,
        )
        for criterion_name_a in profiles_perform.index:
            for criterion_name_b in performance_table.index:
                cv_series = counter_veto_pair(
                    profiles_perform.loc[criterion_name_a],
                    performance_table.loc[criterion_name_b],
                    scales,
                    counter_veto_thresholds,
                    validated=True,
                )
                result_profs_alt[criterion_name_b][criterion_name_a] = [
                    cv_name for cv_name, cv_value in cv_series.items() if cv_value
                ]
        return result_alt_profs, result_profs_alt

    result = pd.DataFrame(
        [[[] * len(performance_table.index)] * len(performance_table.index)],
        index=performance_table.index,
        columns=performance_table.index,
    )
    for criterion_name_a in performance_table.index:
        for criterion_name_b in performance_table.index:
            cv_series = counter_veto_pair(
                performance_table.loc[criterion_name_a],
                performance_table.loc[criterion_name_b],
                scales,
                counter_veto_thresholds,
                validated=True,
            )
            result[criterion_name_b][criterion_name_a] = [
                cv_name for cv_name, cv_value in cv_series.items() if cv_value
            ]
    return result


def counter_veto_count(
    alternatives_perform: pd.DataFrame,
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    counter_veto_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
    profiles_perform: Optional[pd.DataFrame] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """_summary_

    :param alternatives_perform: _description_
    :param scales: _description_
    :param counter_veto_thresholds: _description_
    :param profiles_perform: _description_

    :return: _description_
    """
    _consistent_criteria_names(
        alternatives_perform=alternatives_perform,
        profiles_perform=profiles_perform,
        scales=scales,
        counter_veto_thresholds=counter_veto_thresholds,
    )
    _check_df_index(alternatives_perform, index_type="alternatives")
    _check_df_index(profiles_perform, index_type="profiles")
    if profiles_perform is not None:
        return pd.DataFrame(
            [
                [
                    sum(
                        counter_veto_pair(
                            alternatives_perform.loc[alt_name],
                            profiles_perform.loc[prof_name],
                            scales,
                            counter_veto_thresholds,
                            validated=True,
                        ).values
                    )
                    for prof_name in profiles_perform.index.values
                ]
                for alt_name in alternatives_perform.index.values
            ],
            index=alternatives_perform.index,
            columns=profiles_perform.index,
        ), pd.DataFrame(
            [
                [
                    sum(
                        counter_veto_pair(
                            profiles_perform.loc[prof_name],
                            alternatives_perform.loc[alt_name],
                            scales,
                            counter_veto_thresholds,
                            validated=True,
                        ).values
                    )
                    for alt_name in alternatives_perform.index.values
                ]
                for prof_name in profiles_perform.index.values
            ],
            index=profiles_perform.index,
            columns=alternatives_perform.index,
        )
    return pd.DataFrame(
        [
            [
                sum(
                    counter_veto_pair(
                        alternatives_perform.loc[alt_name_a],
                        alternatives_perform.loc[alt_name_b],
                        scales,
                        counter_veto_thresholds,
                        validated=True,
                    ).values
                )
                for alt_name_b in alternatives_perform.index.values
            ]
            for alt_name_a in alternatives_perform.index.values
        ],
        index=alternatives_perform.index,
        columns=alternatives_perform.index,
    )
