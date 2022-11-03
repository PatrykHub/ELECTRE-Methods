"""This module implements methods to compute discordance."""
from typing import Any, Dict, Optional, Union

import pandas as pd

from ..core.aliases import NumericValue
from ..core.functions import Threshold
from ..core.scales import PreferenceDirection, QuantitativeScale
from ._validate import _both_values_in_scale, _inverse_values


def discordance_bin_marginal(
    a_value: NumericValue,
    b_value: NumericValue,
    scale: QuantitativeScale,
    veto_threshold: Threshold,
    inverse: bool = False,
) -> int:
    """_summary_

    :param a_value: _description_
    :param b_value: _description_
    :param scale: _description_
    :param veto_threshold: _description_
    :param inverse: _description_, defaults to False

    :return: _description_
    """
    _both_values_in_scale(a_value, b_value, scale)
    a_value, b_value, scale = _inverse_values(a_value, b_value, scale, inverse)

    if scale.preference_direction == PreferenceDirection.MAX:
        return 1 if b_value - a_value >= veto_threshold(a_value) else 0
    return 1 if a_value - b_value >= veto_threshold(a_value) else 0


def discordance_bin_comprehensive(
    a_values: Union[Dict[Any, NumericValue], pd.Series],
    b_values: Union[Dict[Any, NumericValue], pd.Series],
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    veto_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
    inverse: bool = False,
) -> int:
    """_summary_

    :param a_values: _description_
    :param b_values: _description_
    :param scales: _description_
    :param veto_thresholds: _description_
    :param inverse: _description_, defaults to False

    :return: _description_
    """
    return (
        1
        if 1
        in [
            discordance_bin_marginal(
                a_values[criterion_name],
                b_values[criterion_name],
                scales[criterion_name],
                veto_thresholds[criterion_name],
                inverse,
            )
            if veto_thresholds[criterion_name] is not None
            else 0
            for criterion_name in a_values.keys()
        ]
        else 0
    )


def discordance_bin(
    alternatives_perform: pd.DataFrame,
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    veto_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
    profiles_perform: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """_summary_

    :param alternatives_perform: _description_
    :param scales: _description_
    :param veto_thresholds: _description_
    :param profiles_perform: _description_

    :return: _description_
    """
    if profiles_perform is not None:
        return pd.DataFrame(
            [
                [
                    discordance_bin_comprehensive(
                        alternatives_perform.loc[alt_name],
                        profiles_perform.loc[prof_name],
                        scales,
                        veto_thresholds,
                    )
                    for prof_name in profiles_perform.index
                ]
                for alt_name in alternatives_perform.index
            ],
            index=alternatives_perform.index,
            columns=profiles_perform.index,
        )
    return pd.DataFrame(
        [
            [
                discordance_bin_comprehensive(
                    alternatives_perform.loc[alt_name_a],
                    alternatives_perform.loc[alt_name_b],
                    scales,
                    veto_thresholds,
                )
                for alt_name_b in alternatives_perform.index
            ]
            for alt_name_a in alternatives_perform.index
        ],
        index=alternatives_perform.index,
        columns=alternatives_perform.index,
    )


def discordance_marginal(
    a_value: NumericValue,
    b_value: NumericValue,
    scale: QuantitativeScale,
    preference_threshold: Threshold,
    veto_threshold: Threshold,
    pre_veto_threshold: Optional[Threshold] = None,
    inverse: bool = False,
) -> NumericValue:
    """_summary_

    :param a_value: _description_
    :param b_value: _description_
    :param scale: _description_
    :param preference_threshold: _description_
    :param veto_threshold: _description_
    :param pre_veto_threshold: _description_
    :param inverse: _description_, defaults to False

    :raises ValueError:

    :return: _description_
    """
    _both_values_in_scale(a_value, b_value, scale)
    a_value, b_value, scale = _inverse_values(a_value, b_value, scale, inverse)

    veto: NumericValue = veto_threshold(a_value)
    preference: NumericValue = preference_threshold(a_value)
    pre_veto: Optional[NumericValue] = (
        pre_veto_threshold(a_value) if pre_veto_threshold is not None else None
    )

    if preference >= veto:
        raise ValueError("Preference threshold must be less than the veto threshold.")
    if pre_veto and (pre_veto >= veto or pre_veto <= preference):
        raise ValueError(
            "Pre-veto must be less than the veto threshold and greater than the preference one."
        )

    if scale.preference_direction == PreferenceDirection.MAX:
        if b_value - a_value > veto:
            return 1
        elif b_value - a_value <= preference:
            return 0
        return (veto - b_value + a_value) / (veto - preference)

    if a_value - b_value > veto:
        return 1
    elif a_value - b_value <= preference:
        return 0
    return (veto - a_value + b_value) / (veto - preference)


def discordance_comprehensive(
    a_values: Union[Dict[Any, NumericValue], pd.Series],
    b_values: Union[Dict[Any, NumericValue], pd.Series],
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    weights: Union[Dict[Any, NumericValue], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    veto_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
    pre_veto_thresholds: Optional[
        Union[Dict[Any, Optional[Threshold]], pd.Series]
    ] = None,
) -> NumericValue:
    """_summary_

    :param a_values: _description_
    :param b_values: _description_
    :param scales: _description_
    :param weights: _description_
    :param preference_thresholds: _description_
    :param veto_thresholds: _description_
    :param pre_veto_thresholds: _description_

    :return: _description_
    """
    return sum(
        [
            weights[criterion_name]
            * discordance_marginal(
                a_values[criterion_name],
                b_values[criterion_name],
                scales[criterion_name],
                preference_thresholds[criterion_name],
                veto_thresholds[criterion_name],
                pre_veto_thresholds[criterion_name] if pre_veto_thresholds else None,
            )
            if veto_thresholds[criterion_name] is not None
            else 0
            for criterion_name in a_values.keys()
        ]
    ) / sum(weights.values() if isinstance(weights, dict) else weights.values)


def discordance(
    alternatives_perform: pd.DataFrame,
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    weights: Union[Dict[Any, NumericValue], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    veto_thresholds: Union[Dict[Any, Threshold], pd.Series],
    pre_veto_thresholds: Optional[
        Union[Dict[Any, Optional[Threshold]], pd.Series]
    ] = None,
    profiles_perform: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """_summary_

    :param alternatives_perform: _description_
    :param scales: _description_
    :param weights: _description_
    :param preference_thresholds: _description_
    :param veto_thresholds: _description_
    :param pre_veto_thresholds: _description_
    :param profiles_perform: _description_

    :return: _description_
    """
    if profiles_perform is not None:
        return pd.DataFrame(
            [
                [
                    discordance_comprehensive(
                        alternatives_perform.loc[alt_name],
                        profiles_perform.loc[prof_name],
                        scales,
                        weights,
                        preference_thresholds,
                        veto_thresholds,
                        pre_veto_thresholds,
                    )
                    for prof_name in profiles_perform.index
                ]
                for alt_name in alternatives_perform.index
            ],
            index=alternatives_perform.index,
            columns=profiles_perform.index,
        )
    return pd.DataFrame(
        [
            [
                discordance_comprehensive(
                    alternatives_perform.loc[alt_name_a],
                    alternatives_perform.loc[alt_name_b],
                    scales,
                    weights,
                    preference_thresholds,
                    veto_thresholds,
                    pre_veto_thresholds,
                )
                for alt_name_b in alternatives_perform.index
            ]
            for alt_name_a in alternatives_perform.index
        ],
        index=alternatives_perform.index,
        columns=alternatives_perform.index,
    )


def counter_veto_marginal(
    a_value: NumericValue,
    b_value: NumericValue,
    scale: QuantitativeScale,
    counter_veto_threshold: Threshold,
) -> bool:
    """_summary_

    :param a_value: _description_
    :param b_value: _description_
    :param scale: _description_
    :param counter_veto_threshold: _description_

    :return: _description_
    """
    _both_values_in_scale(a_value, b_value, scale)
    return (
        b_value - a_value > counter_veto_threshold(a_value)
        if scale.preference_direction == PreferenceDirection.MAX
        else a_value - b_value > counter_veto_threshold(a_value)
    )


def counter_veto_pair(
    a_values: Union[Dict[Any, NumericValue], pd.Series],
    b_values: Union[Dict[Any, NumericValue], pd.Series],
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    counter_veto_thresholds: Union[Dict[Any, Threshold], pd.Series],
) -> pd.Series:
    """_summary_

    :param a_values: _description_
    :param b_values: _description_
    :param scales: _description_
    :param counter_veto_thresholds: _description_

    :return: _description_
    """
    return pd.Series(
        [
            counter_veto_marginal(
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
    alt_values: Union[Dict[Any, NumericValue], pd.Series],
    performance_table: pd.DataFrame,
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    counter_veto_thresholds: Union[Dict[Any, Threshold], pd.Series],
) -> pd.DataFrame:
    """_summary_

    :param alt_values: _description_
    :param performance_table: _description_
    :param scales: _description_
    :param counter_veto_thresholds: _description_

    :return: _description_
    """
    return pd.DataFrame(
        [
            counter_veto_pair(
                alt_values,
                performance_table.loc[alt_name],
                scales,
                counter_veto_thresholds,
            )
            for alt_name in performance_table.index
        ],
        index=performance_table.index,
    )


def counter_veto_all(
    alternatives_perform: pd.DataFrame,
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    counter_veto_thresholds: Union[Dict[Any, Threshold], pd.Series],
) -> pd.Series:
    """_summary_

    :param alternatives_perform: _description_
    :param scales: _description_
    :param counter_veto_thresholds: _description_

    :return: _description_
    """
    return pd.Series(
        [
            counter_veto(
                alternatives_perform.loc[name],
                alternatives_perform,
                scales,
                counter_veto_thresholds,
            )
            for name in alternatives_perform.index
        ],
        index=alternatives_perform.index,
    )


def counter_veto_count(
    alternatives_perform: pd.DataFrame,
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    counter_veto_thresholds: Union[Dict[Any, Threshold], pd.Series],
    profiles_perform: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """_summary_

    :param alternatives_perform: _description_
    :param scales: _description_
    :param counter_veto_thresholds: _description_
    :param profiles_perform: _description_

    :return: _description_
    """
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
                        ).values
                    )
                    for prof_name in profiles_perform.index
                ]
                for alt_name in alternatives_perform.index
            ],
            index=alternatives_perform.index,
            columns=profiles_perform.index,
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
                    ).values
                )
                for alt_name_b in alternatives_perform.index
            ]
            for alt_name_a in alternatives_perform.index
        ],
        index=alternatives_perform.index,
        columns=alternatives_perform.index,
    )
