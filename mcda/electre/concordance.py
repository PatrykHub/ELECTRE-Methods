"""This module implements methods to compute concordance."""
from typing import Any, Dict, Optional, Tuple, Union, get_args
from enum import Enum

import pandas as pd

from ._validate import (
    _all_lens_equal,
    _both_values_in_scale,
    _inverse_values,
    _weights_proper_vals,
)
from ..core.aliases import NumericValue
from ..core.functions import Threshold
from ..core.scales import PreferenceDirection, QuantitativeScale


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
        raise ValueError(
            "Indifference threshold can't be bigger than the preference threshold."
        )

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
                inverse
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
) -> pd.DataFrame:
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


def concordance_profiles(
        alternatives_perform: pd.DataFrame,
        scales: Union[Dict[Any, QuantitativeScale], pd.Series],
        weights: Union[Dict[Any, NumericValue], pd.Series],
        indifference_thresholds: pd.DataFrame,
        preference_thresholds: pd.DataFrame,
        profiles_perform: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param alternatives_perform:
    :param scales:
    :param weights:
    :param indifference_thresholds:
    :param preference_thresholds:
    :param profiles_perform:

    :return:
    """
    return pd.DataFrame(
        [
            [
                concordance_comprehensive(
                    alternatives_perform.loc[alt_name],
                    profiles_perform.loc[prof_name],
                    scales,
                    weights,
                    indifference_thresholds.loc[prof_name],
                    preference_thresholds.loc[prof_name],
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
                    indifference_thresholds.loc[prof_name],
                    preference_thresholds.loc[prof_name],
                )
                for alt_name in alternatives_perform.index
            ]
            for prof_name in profiles_perform.index
        ],
        index=profiles_perform.index,
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
        a_value - b_value > reinforcement_threshold(a_value)
        if scale.preference_direction == PreferenceDirection.MIN
        else a_value - b_value < reinforcement_threshold(a_value)
    )


def concordance_reinforced_comprehensive(
        a_values: Union[Dict[Any, NumericValue], pd.Series],
        b_values: Union[Dict[Any, NumericValue], pd.Series],
        scales: Union[Dict[Any, QuantitativeScale], pd.Series],
        weights: Union[Dict[Any, NumericValue], pd.Series],
        indifference_thresholds: Union[Dict[Any, Threshold], pd.Series],
        preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
        reinforced_thresholds: Union[Dict[Any, Threshold], pd.Series],
        reinforcement_factors: Union[Dict[Any, NumericValue], pd.Series],
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
    reinforce_occur = pd.Series(
        [
            is_reinforcement_occur(
                a_values[criterion_name],
                b_values[criterion_name],
                scales[criterion_name],
                reinforced_thresholds[criterion_name],
            )
            for criterion_name in a_values.keys()
        ],
        index=list(a_values.keys()),
    )

    sum_weights_thresholds = sum(
        [
            weights[criterion_name]
            * reinforce_occur[criterion_name]
            * reinforcement_factors[criterion_name]
            for criterion_name in reinforce_occur
        ]
    )

    return (
       sum_weights_thresholds
       + sum(
            [
                0
                if reinforce_occur[criterion_name]
                else weights[criterion_name]
                * concordance_marginal(
                   a_values[criterion_name],
                   b_values[criterion_name],
                   scales[criterion_name],
                   indifference_thresholds[criterion_name],
                   preference_thresholds[criterion_name],
                   inverse,
                )
                for criterion_name in reinforce_occur.keys()
            ]
       )
    ) / (
       sum_weights_thresholds
       + sum(
           [
               weights[criterion_name] * (not reinforce_occur[criterion_name])
               for criterion_name in reinforce_occur.keys()
           ]
       )
    )


def concordance_reinforced(
        alternatives_perform: pd.DataFrame,
        scales: Union[Dict[Any, QuantitativeScale], pd.Series],
        weights: Union[Dict[Any, NumericValue], pd.Series],
        indifference_thresholds: Union[Dict[Any, Threshold], pd.Series],
        preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
        reinforced_thresholds: Union[Dict[Any, Threshold], pd.Series],
        reinforcement_factors: Union[Dict[Any, NumericValue], pd.Series],
        profiles_perform: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
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


def concordance_reinforced_profiles(
        alternatives_perform: pd.DataFrame,
        scales: Union[Dict[Any, QuantitativeScale], pd.Series],
        weights: Union[Dict[Any, NumericValue], pd.Series],
        indifference_thresholds: pd.DataFrame,
        preference_thresholds: pd.DataFrame,
        reinforced_thresholds: pd.DataFrame,
        reinforcement_factors: Union[Dict[Any, NumericValue], pd.Series],
        profiles_perform: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    return pd.DataFrame(
        [
            [
                concordance_reinforced_comprehensive(
                    alternatives_perform.loc[alt_name],
                    profiles_perform.loc[prof_name],
                    scales,
                    weights,
                    indifference_thresholds.loc[prof_name],
                    preference_thresholds.loc[prof_name],
                    reinforced_thresholds.loc[prof_name],
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
                    indifference_thresholds.loc[prof_name],
                    preference_thresholds.loc[prof_name],
                    reinforced_thresholds.loc[prof_name],
                    reinforcement_factors,
                )
                for alt_name in alternatives_perform.index
            ]
            for prof_name in profiles_perform.index
        ],
        index=profiles_perform.index,
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
    def __init__(self, interaction_type: Optional[InteractionType] = None,
                 function_type: Optional[FunctionType] = None,
                 factor: NumericValue = None):
        self.interaction = interaction_type
        self.function_type = function_type
        self.factor = factor

    def __repr__(self) -> str:
        """Return instance representation string."""
        return (
            f"{self.interaction.value} |"
            f" {self.function_type.value} |"
            f" {self.factor} "
        )


def concordance_with_interactions_marginal(
        a_values: Union[Dict[Any, NumericValue], pd.Series],
        b_values: Union[Dict[Any, NumericValue], pd.Series],
        scales: Union[Dict[Any, QuantitativeScale], pd.Series],
        weights: Union[Dict[Any, NumericValue], pd.Series],
        indifference_thresholds: Union[Dict[Any, Threshold], pd.Series],
        preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
        interactions: pd.DataFrame
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
    _all_lens_equal(
        a=a_values,
        b=b_values,
        scales=scales,
        weights=weights,
        indifference_thresholds=indifference_thresholds,
        preference_thresholds=preference_thresholds,
    )
    _weights_proper_vals(weights)

    for i in interactions.index:
        if interactions[i][i] is not None:
            raise ValueError("Criterion cannot interact with itself.")
        for j in interactions.columns:
            if interactions[i][j] is not None \
                    and interactions[i][j].interaction not in [
                InteractionType.MW,
                InteractionType.MS,
                InteractionType.A,
            ]:
                raise ValueError(
                    "The interaction type has to be represented "
                    "by one of the following enumeration tokens:\n"
                    "'MW' - Mutual Weakening\n'MS' - Mutual Strengthening\n'A' - Antagonistic"
                )
            if interactions[i][j] is not None and interactions[i][j].function_type not in [
                FunctionType.MIN,
                FunctionType.MUL
            ]:
                raise ValueError(
                    "The Z function has to be represented "
                    "by one of the following enumeration tokens:\n'min' - "
                    "minimum\n'multi' - multiplication"
                )
            if interactions[i][j] is not None and not isinstance(
                    interactions[i][j].factor, get_args(NumericValue)
            ):
                raise TypeError("Interaction factor must be a numerical value.")
            if (
                    interactions[i][j] is not None
                    and interactions[i][j].interaction == InteractionType.MW
                    and weights[i] - abs(interactions[i][j].factor) < 0
            ):
                raise ValueError("Incorrect interaction factor.")
            if (
                    interactions[i][j]
                    and interactions[i][j].interaction == InteractionType.A
                    and weights[i] - interactions[i][j].factor < 0
            ):
                raise ValueError("Incorrect interaction factor.")

    mutual_strengthening = []
    mutual_weakening = []
    antagonistic = []
    marginal_concordances = [
        concordance_marginal(
            a_values[criterion_name],
            b_values[criterion_name],
            scales[criterion_name],
            indifference_thresholds[criterion_name],
            preference_thresholds[criterion_name]
        )
        for criterion_name in a_values.keys()
    ]
    for i in interactions.index:
        for j in interactions.columns:
            if interactions[i][j] is not None:
                c_i = marginal_concordances[i]
                c_j = marginal_concordances[j]
                if interactions[i][j].interaction == InteractionType.MS:
                    strengthening_weight = (
                            weights[i] + weights[j] + interactions[i][j].factor
                    )
                    mutual_strengthening.append(
                        strengthening_weight * min(c_i, c_j)
                        if interactions[i][j].function_type == FunctionType.MIN
                        else strengthening_weight * c_i * c_j
                    )
                elif interactions[i][j].interaction == InteractionType.MW:
                    weakening_weight = weights[i] + weights[j] + interactions[i][j].factor
                    mutual_weakening.append(
                        weakening_weight * min(c_i, c_j)
                        if interactions[i][j].function_type == FunctionType.MIN
                        else weakening_weight * c_i * c_j
                    )
                else:
                    antagonistic_weight = (
                            weights[i] + weights[j] - interactions[i][j].factor
                    )
                    antagonistic.append(
                        antagonistic_weight * min(c_i, c_j)
                        if interactions[i][j].function_type == FunctionType.MIN
                        else antagonistic_weight * c_i * c_j
                    )

    interaction_sum = sum(mutual_strengthening) + sum(mutual_weakening) - sum(antagonistic)

    return (
                   sum(
                       [
                           weights[criterion_name]
                           * marginal_concordances[criterion_name]
                           for criterion_name in a_values.keys()
                       ]
                   )
                   + interaction_sum
           ) / (
                   sum(weights)
                   + interaction_sum
           )


def concordance_with_interactions(
        alternatives_perform: pd.DataFrame,
        scales: Union[Dict[Any, QuantitativeScale], pd.Series],
        weights: Union[Dict[Any, NumericValue], pd.Series],
        indifference_thresholds: Union[Dict[Any, Threshold], pd.Series],
        preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
        interactions: pd.DataFrame,
        profiles_perform: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
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
                        interactions
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
                concordance_with_interactions_marginal(
                    alternatives_perform.loc[alt_name_a],
                    alternatives_perform.loc[alt_name_b],
                    scales,
                    weights,
                    indifference_thresholds,
                    preference_thresholds,
                    interactions
                )
                for alt_name_b in alternatives_perform.index
            ]
            for alt_name_a in alternatives_perform.index
        ],
        index=alternatives_perform.index,
        columns=alternatives_perform.index,
    )
