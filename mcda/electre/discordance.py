"""This module implements methods to compute discordance."""
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Union

import pandas as pd

from ..core.aliases import NumericValue
from ..core.functions import Threshold
from ..core.scales import PreferenceDirection, QuantitativeScale
from ._validate import _both_values_in_scale, _inverse_values


def discordance_bin_marginal(
    a_value: NumericValue,
    b_value: NumericValue,
    scale: QuantitativeScale,
    veto_threshold: Optional[Threshold],
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

    if veto_threshold is None:
        return 0

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
) -> pd.Series:
    """Returns discordance marginals for all criteria between
    two alternatives."""
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
    """_summary_

    :param alternatives_perform: _description_
    :param scales: _description_
    :param veto_thresholds: _description_
    :param profiles_perform: _description_

    :return: _description_
    """
    discordance_function: Callable = discordance_bin_comprehensive

    if return_marginals:
        discordance_function = discordance_bin_criteria_marginals

    if profiles_perform is not None:
        return pd.DataFrame(
            [
                [
                    discordance_function(
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
        ), pd.DataFrame(
            [
                [
                    discordance_function(
                        profiles_perform.loc[prof_name],
                        alternatives_perform.loc[alt_name],
                        scales,
                        veto_thresholds,
                    )
                    for alt_name in alternatives_perform.index
                ]
                for prof_name in profiles_perform.index
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
    veto_threshold: Optional[Threshold],
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

    if veto_threshold is None:
        return 0

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

    discordance_beginning = preference if pre_veto is None else pre_veto
    if scale.preference_direction == PreferenceDirection.MAX:
        if b_value - a_value > veto:
            return 1
        elif b_value - a_value <= discordance_beginning:
            return 0
        return (b_value - a_value - discordance_beginning) / (
            veto - discordance_beginning
        )

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
                (
                    pre_veto_thresholds[criterion_name]
                    if pre_veto_thresholds is not None
                    else None
                ),
            )
            for criterion_name in a_values.keys()
        ]
    ) / sum(weights.values() if isinstance(weights, dict) else weights.values)


def discordance_criteria_marginals(
    a_values: Union[Dict[Any, NumericValue], pd.Series],
    b_values: Union[Dict[Any, NumericValue], pd.Series],
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    veto_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
    pre_veto_thresholds: Optional[
        Union[Dict[Any, Optional[Threshold]], pd.Series]
    ] = None,
) -> pd.Series:
    """Returns discordance marginals for all criteria between
    two alternatives."""
    return pd.Series(
        [
            discordance_marginal(
                a_values[criterion_name],
                b_values[criterion_name],
                scales[criterion_name],
                preference_thresholds[criterion_name],
                veto_thresholds[criterion_name],
                (
                    pre_veto_thresholds[criterion_name]
                    if pre_veto_thresholds is not None
                    else None
                ),
            )
            for criterion_name in a_values.keys()
        ],
        index=list(a_values.keys()),
    )


def discordance(
    alternatives_perform: pd.DataFrame,
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    weights: Union[Dict[Any, NumericValue], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    veto_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
    pre_veto_thresholds: Optional[
        Union[Dict[Any, Optional[Threshold]], pd.Series]
    ] = None,
    profiles_perform: Optional[pd.DataFrame] = None,
    return_marginals: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
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
    discordance_function: Callable = discordance_comprehensive

    if return_marginals:
        discordance_function = discordance_criteria_marginals

    discordance_func_args = (
        (scales, preference_thresholds, veto_thresholds, pre_veto_thresholds)
        if return_marginals
        else (
            scales,
            weights,
            preference_thresholds,
            veto_thresholds,
            pre_veto_thresholds,
        )
    )

    if profiles_perform is not None:
        return pd.DataFrame(
            [
                [
                    discordance_function(
                        alternatives_perform.loc[alt_name],
                        profiles_perform.loc[prof_name],
                        *discordance_func_args
                    )
                    for prof_name in profiles_perform.index
                ]
                for alt_name in alternatives_perform.index
            ],
            index=alternatives_perform.index,
            columns=profiles_perform.index,
        ), pd.DataFrame(
            [
                [
                    discordance_function(
                        profiles_perform.loc[prof_name],
                        alternatives_perform.loc[alt_name],
                        *discordance_func_args
                    )
                    for alt_name in alternatives_perform.index
                ]
                for prof_name in profiles_perform.index
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
                    *discordance_func_args
                )
                for alt_name_b in alternatives_perform.index
            ]
            for alt_name_a in alternatives_perform.index
        ],
        index=alternatives_perform.index,
        columns=alternatives_perform.index,
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
    if (
        non_discordance_type == NonDiscordanceType.DM
        or concordance_comprehensive is None
    ):
        return 1 - max(criteria_discordance_marginals)

    non_discordance = 1
    for index in criteria_discordance_marginals.index:
        if (
            non_discordance_type == NonDiscordanceType.DC
            and criteria_discordance_marginals[index] > concordance_comprehensive
        ):
            non_discordance *= (1 - criteria_discordance_marginals[index]) / (
                1 - concordance_comprehensive
            )

        elif non_discordance_type == NonDiscordanceType.D:
            non_discordance *= 1 - criteria_discordance_marginals[index]

    return non_discordance


def non_discordance(
    criteria_discordance_marginals: pd.DataFrame,
    non_discordance_type: NonDiscordanceType = NonDiscordanceType.DC,
    concordance_comprehensive: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            [
                non_discordance_marginal(
                    criteria_discordance_marginals.loc[alt_name_a, alt_name_b],
                    non_discordance_type,
                    concordance_comprehensive.loc[alt_name_a, alt_name_b]
                    if concordance_comprehensive is not None
                    else None,
                )
                for alt_name_b in criteria_discordance_marginals.columns
            ]
            for alt_name_a in criteria_discordance_marginals.index
        ],
        index=criteria_discordance_marginals.index,
        columns=criteria_discordance_marginals.columns,
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

    return (
        a_value - b_value > counter_veto_threshold(b_value)
        if scale.preference_direction == PreferenceDirection.MAX
        else b_value - a_value > counter_veto_threshold(b_value)
    )


def counter_veto_pair(
    a_values: Union[Dict[Any, NumericValue], pd.Series],
    b_values: Union[Dict[Any, NumericValue], pd.Series],
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    counter_veto_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
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
    if profiles_perform is not None:
        result_alt_profs = pd.DataFrame(
            [[[] * len(performance_table.index)] * len(profiles_perform.index)],
            index=performance_table.index,
            columns=profiles_perform.index,
        )
        for criterion_name_a in performance_table.index:
            for criterion_name_b in profiles_perform.index:
                cv_series = counter_veto_pair(
                    performance_table.loc[criterion_name_a],
                    profiles_perform.loc[criterion_name_b],
                    scales,
                    counter_veto_thresholds,
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
        ), pd.DataFrame(
            [
                [
                    sum(
                        counter_veto_pair(
                            profiles_perform.loc[prof_name],
                            alternatives_perform.loc[alt_name],
                            scales,
                            counter_veto_thresholds,
                        ).values
                    )
                    for alt_name in alternatives_perform.index
                ]
                for prof_name in profiles_perform.index
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
                    ).values
                )
                for alt_name_b in alternatives_perform.index
            ]
            for alt_name_a in alternatives_perform.index
        ],
        index=alternatives_perform.index,
        columns=alternatives_perform.index,
    )
