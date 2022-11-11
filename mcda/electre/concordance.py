"""This module implements methods to compute concordance."""
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from mcda.core.aliases import NumericValue
from mcda.core.functions import Threshold
from mcda.core.scales import PreferenceDirection, QuantitativeScale
from mcda.electre._validate import _both_values_in_scale, _inverse_values


def concordance_marginal(
    a_value: NumericValue,
    b_value: NumericValue,
    scale: QuantitativeScale,
    indifference_threshold: Threshold,
    preference_threshold: Threshold,
    inverse: bool = False,
) -> NumericValue:
    """Computes marginal concordance index c(a, b) ∈ [0, 1],
    which represents a degree to which criterion supports
    the hypothesis about outranking a over b (aSb).

    :param a_value: criterion value
    :param b_value: criterion value
    :param scale: criterion scale with specified preference direction
    :param indifference_threshold: criterion indifference threshold
    :param preference_threshold: criterion preference threshold
    :param inverse: if ``True``, then function will calculate c(b, a)
    with opposite scale preference direction, defaults to ``False``

    :raises ValueError:
        * if the preference threshold is less than the indifference one

    :return: marginal concordance index, value from the [0, 1] interval
    """
    _both_values_in_scale(a_value, b_value, scale)
    a_value, b_value, scale = _inverse_values(a_value, b_value, scale, inverse)

    q_a: NumericValue = indifference_threshold(a_value)
    p_a: NumericValue = preference_threshold(a_value)

    if q_a >= p_a:
        raise ValueError("Indifference threshold can't be bigger than the preference threshold.")

    if scale.preference_direction == PreferenceDirection.MIN:
        if a_value - b_value <= q_a:
            return 1.0
        elif a_value - b_value > p_a:
            return 0.0
        return (b_value - a_value + p_a) / (p_a - q_a)

    if b_value - a_value <= q_a:
        return 1.0
    elif b_value - a_value > p_a:
        return 0.0
    return (p_a - (b_value - a_value)) / (p_a - q_a)


def concordance_comprehensive(
    a_values: Union[Dict[Any, NumericValue], pd.Series],
    b_values: Union[Dict[Any, NumericValue], pd.Series],
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    weights: Union[Dict[Any, NumericValue], pd.Series],
    indifference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    inverse: bool = False,
) -> NumericValue:
    """_summary_

    :param a_values: _description_
    :param b_values: _description_
    :param scales: _description_
    :param weights: _description_
    :param indifference_thresholds: _description_
    :param preference_thresholds: _description_
    :param inverse: _description_, defaults to False

    :return: _description_
    """
    return sum(
        [
            weights[criterion_name]
            * concordance_marginal(
                a_values[criterion_name],
                b_values[criterion_name],
                scales[criterion_name],
                indifference_thresholds[criterion_name],
                preference_thresholds[criterion_name],
                inverse,
            )
            for criterion_name in a_values.keys()
        ]
    ) / sum(weights.values() if isinstance(weights, dict) else weights.values)


def concordance(
    alternatives_perform: pd.DataFrame,
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    weights: Union[Dict[Any, NumericValue], pd.Series],
    indifference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    profiles_perform: Optional[pd.DataFrame] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """_summary_

    :param alternatives_perform: _description_
    :param scales: _description_
    :param weights: _description_
    :param indifference_thresholds: _description_
    :param preference_thresholds: _description_
    :param profiles_perform: _description_

    :return: _description_
    """
    if profiles_perform is not None:
        return pd.DataFrame(
            [
                [
                    concordance_comprehensive(
                        alternatives_perform.loc[alt_name],
                        profiles_perform.loc[prof_name],
                        scales,
                        weights,
                        indifference_thresholds,
                        preference_thresholds,
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
                    concordance_comprehensive(
                        profiles_perform.loc[prof_name],
                        alternatives_perform.loc[alt_name],
                        scales,
                        weights,
                        indifference_thresholds,
                        preference_thresholds,
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
                concordance_comprehensive(
                    alternatives_perform.loc[alt_name_a],
                    alternatives_perform.loc[alt_name_b],
                    scales,
                    weights,
                    indifference_thresholds,
                    preference_thresholds,
                )
                for alt_name_b in alternatives_perform.index
            ]
            for alt_name_a in alternatives_perform.index
        ],
        index=alternatives_perform.index,
        columns=alternatives_perform.index,
    )


def is_reinforcement_occur(
    a_value: NumericValue,
    b_value: NumericValue,
    scale: QuantitativeScale,
    reinforcement_threshold: Threshold,
) -> bool:
    """_summary_

    :param a_value: _description_
    :param b_value: _description_
    :param scale: _description_
    :param reinforcement_threshold: _description_

    :return: _description_
    """
    return (
        a_value - b_value > reinforcement_threshold(b_value)
        if scale.preference_direction == PreferenceDirection.MAX
        else b_value - a_value > reinforcement_threshold(b_value)
    )


def _get_reinforced_criteria(
    a_values: Union[Dict[Any, NumericValue], pd.Series],
    b_values: Union[Dict[Any, NumericValue], pd.Series],
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    reinforced_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
) -> pd.Series:
    """Return a series with information about reinforcement."""
    result = pd.Series([False] * len(a_values), index=list(a_values.keys()))
    for criterion_name in a_values.keys():
        threshold = reinforced_thresholds[criterion_name]
        if isinstance(threshold, Threshold):
            result[criterion_name] = is_reinforcement_occur(
                a_values[criterion_name],
                b_values[criterion_name],
                scales[criterion_name],
                threshold,
            )
    return result


def concordance_reinforced_comprehensive(
    a_values: Union[Dict[Any, NumericValue], pd.Series],
    b_values: Union[Dict[Any, NumericValue], pd.Series],
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    weights: Union[Dict[Any, NumericValue], pd.Series],
    indifference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    reinforced_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
    reinforcement_factors: Union[Dict[Any, Optional[NumericValue]], pd.Series],
    inverse: bool = False,
) -> NumericValue:
    """_summary_

    :param a_values: _description_
    :param b_values: _description_
    :param scales: _description_
    :param weights: _description_
    :param indifference_thresholds: _description_
    :param preference_thresholds: _description_
    :param reinforced_thresholds: _description_
    :param reinforcement_factors: _description_
    :param inverse: _description_, defaults to False

    :return: _description_
    """
    reinforce_occur = _get_reinforced_criteria(a_values, b_values, scales, reinforced_thresholds)
    sum_weights_reinforced: NumericValue = 0
    sum_weights_not_reinforced: NumericValue = 0
    sum_concordances_not_reinforced: NumericValue = 0
    for criterion_name in reinforce_occur.index:
        if reinforce_occur[criterion_name]:
            factor = reinforcement_factors[criterion_name]
            if factor is None:
                raise ValueError(
                    "Reinforce threshold was provided, but there's missing factor "
                    f"on {criterion_name} criterion."
                )
            else:
                sum_weights_reinforced += weights[criterion_name] * factor
        else:
            sum_weights_not_reinforced += weights[criterion_name]
            sum_concordances_not_reinforced += weights[criterion_name] * concordance_marginal(
                a_values[criterion_name],
                b_values[criterion_name],
                scales[criterion_name],
                indifference_thresholds[criterion_name],
                preference_thresholds[criterion_name],
                inverse,
            )

    return (sum_concordances_not_reinforced + sum_weights_reinforced) / (
        sum_weights_reinforced + sum_weights_not_reinforced
    )


def concordance_reinforced(
    alternatives_perform: pd.DataFrame,
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    weights: Union[Dict[Any, NumericValue], pd.Series],
    indifference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    reinforced_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
    reinforcement_factors: Union[Dict[Any, Optional[NumericValue]], pd.Series],
    profiles_perform: Optional[pd.DataFrame] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """_summary_

    :param alternatives_perform: _description_
    :param scales: _description_
    :param weights: _description_
    :param indifference_thresholds: _description_
    :param preference_thresholds: _description_
    :param reinforced_thresholds: _description_
    :param reinforcement_factors: _description_
    :param profiles_perform: _description_

    :return: _description_
    """
    if profiles_perform is not None:
        return pd.DataFrame(
            [
                [
                    concordance_reinforced_comprehensive(
                        alternatives_perform.loc[alt_name],
                        profiles_perform.loc[prof_name],
                        scales,
                        weights,
                        indifference_thresholds,
                        preference_thresholds,
                        reinforced_thresholds,
                        reinforcement_factors,
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
                    concordance_reinforced_comprehensive(
                        profiles_perform.loc[prof_name],
                        alternatives_perform.loc[alt_name],
                        scales,
                        weights,
                        indifference_thresholds,
                        preference_thresholds,
                        reinforced_thresholds,
                        reinforcement_factors,
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
                concordance_reinforced_comprehensive(
                    alternatives_perform.loc[alt_name_a],
                    alternatives_perform.loc[alt_name_b],
                    scales,
                    weights,
                    indifference_thresholds,
                    preference_thresholds,
                    reinforced_thresholds,
                    reinforcement_factors,
                )
                for alt_name_b in alternatives_perform.index
            ]
            for alt_name_a in alternatives_perform.index
        ],
        index=alternatives_perform.index,
        columns=alternatives_perform.index,
    )


class InteractionType(Enum):
    MS = "Mutual Strengthening"
    MW = "Mutual Weakening"
    A = "Antagonistic"


class FunctionType(Enum):
    MUL = "Multiplication"
    MIN = "Minimisation"


class Interaction:
    def __init__(
        self,
        interaction_type: InteractionType,
        factor: NumericValue,
    ):
        self.interaction = interaction_type
        self.factor = factor

    def __repr__(self) -> str:
        """Return instance representation string."""
        return f"{self.interaction.value} |" f" {self.factor} "


def concordance_with_interactions_marginal(
    a_values: Union[Dict[Any, NumericValue], pd.Series],
    b_values: Union[Dict[Any, NumericValue], pd.Series],
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    weights: Union[Dict[Any, NumericValue], pd.Series],
    indifference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    interactions: pd.DataFrame,
    function_type: FunctionType = FunctionType.MIN,
) -> NumericValue:
    """

    :param interactions:
    :param a_values:
    :param b_values:
    :param scales:
    :param weights:
    :param indifference_thresholds:
    :param preference_thresholds:
    :return:
    """
    mutual, antagonistic = [], []
    marginal_concordances = pd.Series(
        [
            concordance_marginal(
                a_values[criterion_name],
                b_values[criterion_name],
                scales[criterion_name],
                indifference_thresholds[criterion_name],
                preference_thresholds[criterion_name],
            )
            for criterion_name in a_values.keys()
        ],
        index=list(a_values.keys()),
    )

    interaction_value = (
        lambda c_i, c_j: min(c_i, c_j) if function_type == FunctionType.MIN else c_i * c_j
    )

    for i in interactions.index:
        for j in interactions.columns:
            if interactions[j][i] is not None:
                c_i = marginal_concordances[i]
                c_j = marginal_concordances[j]

                if interactions[j][i].interaction in (
                    InteractionType.MS,
                    InteractionType.MW,
                ):
                    mutual.append(interaction_value(c_i, c_j) * interactions[j][i].factor)

                else:
                    antagonistic.append(
                        interaction_value(
                            c_i,
                            concordance_marginal(
                                b_values[j],
                                a_values[j],
                                scales[j],
                                indifference_thresholds[j],
                                preference_thresholds[j],
                            ),
                        )
                        * interactions[j][i].factor
                    )

    interaction_sum = sum(mutual) - sum(antagonistic)

    return (
        sum(
            [
                weights[criterion_name] * marginal_concordances[criterion_name]
                for criterion_name in a_values.keys()
            ]
        )
        + interaction_sum
    ) / (sum(weights) + interaction_sum)


def concordance_with_interactions(
    alternatives_perform: pd.DataFrame,
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    weights: Union[Dict[Any, NumericValue], pd.Series],
    indifference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    interactions: pd.DataFrame,
    function_type: FunctionType = FunctionType.MIN,
    profiles_perform: Optional[pd.DataFrame] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """_summary_

    :param alternatives_perform: _description_
    :param scales: _description_
    :param weights: _description_
    :param indifference_thresholds: _description_
    :param preference_thresholds: _description_
    :param interactions: _description_
    :param profiles_perform: _description_

    :return: _description_
    """
    if profiles_perform is not None:
        return pd.DataFrame(
            [
                [
                    concordance_with_interactions_marginal(
                        alternatives_perform.loc[alt_name],
                        profiles_perform.loc[prof_name],
                        scales,
                        weights,
                        indifference_thresholds,
                        preference_thresholds,
                        interactions,
                        function_type,
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
                    concordance_with_interactions_marginal(
                        profiles_perform.loc[prof_name],
                        alternatives_perform.loc[alt_name],
                        scales,
                        weights,
                        indifference_thresholds,
                        preference_thresholds,
                        interactions,
                        function_type,
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
                concordance_with_interactions_marginal(
                    alternatives_perform.loc[alt_name_a],
                    alternatives_perform.loc[alt_name_b],
                    scales,
                    weights,
                    indifference_thresholds,
                    preference_thresholds,
                    interactions,
                    function_type,
                )
                for alt_name_b in alternatives_perform.index
            ]
            for alt_name_a in alternatives_perform.index
        ],
        index=alternatives_perform.index,
        columns=alternatives_perform.index,
    )
