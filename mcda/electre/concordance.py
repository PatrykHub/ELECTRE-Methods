"""This module implements methods to compute concordance."""
from typing import Any, Dict, List, Optional, Tuple, Union, get_args

import pandas as pd

from ..core.aliases import NumericValue
from ..core.functions import Threshold
from ..core.scales import PreferenceDirection, QuantitativeScale
from ._validate import (
    _all_lens_equal,
    _both_values_in_scale,
    _inverse_values,
    _weights_proper_vals,
)


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
    :param indifference_threshold:
    :param preference_threshold:
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
            for criterion_name in reinforce_occur.keys()
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


def concordance_with_interactions(
    a: List[NumericValue],
    b: List[NumericValue],
    scales: List[QuantitativeScale],
    weights: List[NumericValue],
    interactions: List[List[List]],  # missed typing
    indifference_thresholds: List[Threshold],
    preference_thresholds: List[Threshold],
) -> NumericValue:
    """
    :param a:
    :param b:
    :param scales:
    :param weights:
    :param interactions:
    :param indifferenceThreshold:
    :param preferenceThreshold:
    :return:
    """
    _all_lens_equal(
        a=a,
        b=b,
        scales=scales,
        weights=weights,
        indifference_thresholds=indifference_thresholds,
        preference_thresholds=preference_thresholds,
    )
    _weights_proper_vals(weights)

    if not isinstance(interactions, list) and not all(
        1 if isinstance(item, list) else 0 in interactions for item in interactions
    ):
        raise TypeError("Interactions have to be represented as a matrix.")
    if not len(interactions) == len(a) and not all(
        1 if len(row) == len(a) else 0 in interactions for row in interactions
    ):
        raise ValueError("Interactions have to be a square matrix.")
    for i in range(len(interactions)):
        if interactions[i][i] != []:
            raise ValueError("Criterion cannot interact with itself.")
        for j in range(len(interactions[i])):
            if len(interactions[i][j]) > 0 and len(interactions[i][j]) != 3:
                raise ValueError(
                    "Each interaction has to be represented as a list of length 3."
                )
            if len(interactions[i][j]) == 3 and interactions[i][j][0] not in [
                "MW",
                "MS",
                "A",
            ]:
                raise ValueError(
                    "The interaction type has to be represented by one of the following tokens:\n"
                    "'MW' - Mutual Weakening\n'MS' - Mutual Strengthening\n'A' - Antagonistic"
                )
            if len(interactions[i][j]) == 3 and interactions[i][j][1] not in [
                "min",
                "multi",
            ]:
                raise ValueError(
                    "The Z function has to be represented by one of the following tokens:\n'min' - "
                    "minimum\n'multi' - multiplication"
                )
            if len(interactions[i][j]) == 3 and not isinstance(
                interactions[i][j][2], get_args(NumericValue)
            ):
                raise TypeError("Interaction factor must be a numerical value.")
            if (
                len(interactions[i][j]) == 3
                and interactions[i][j][0] == "MW"
                and weights[i] - abs(interactions[i][j][2]) < 0
            ):
                raise ValueError("Incorrect interaction factor.")
            if (
                len(interactions[i][j]) == 3
                and interactions[i][j][0] == "A"
                and weights[i] - interactions[i][j][2] < 0
            ):
                raise ValueError("Incorrect interaction factor.")

    mutual_strengthening = []
    mutual_weakening = []
    antagonistic = []
    for i in range(len(interactions)):
        for j in range(len(interactions[i])):
            if len(interactions[i][j]) > 1:
                c_i = concordance_marginal(
                    a[i],
                    b[i],
                    scales[i],
                    indifference_thresholds[i],
                    preference_thresholds[i],
                )
                c_j = concordance_marginal(
                    a[j],
                    b[j],
                    scales[j],
                    indifference_thresholds[j],
                    preference_thresholds[j],
                )
                if interactions[i][j][0] == "MS":
                    strengthening_weight = (
                        weights[i] + weights[j] + interactions[i][j][2]
                    )
                    mutual_strengthening.append(
                        strengthening_weight * min(c_i, c_j)
                        if interactions[i][j][1] == "min"
                        else strengthening_weight * c_i * c_j
                    )
                elif interactions[i][j][0] == "MW":
                    weakening_weight = weights[i] + weights[j] + interactions[i][j][2]
                    mutual_weakening.append(
                        weakening_weight * min(c_i, c_j)
                        if interactions[i][j][1] == "min"
                        else weakening_weight * c_i * c_j
                    )
                else:
                    antagonistic_weight = (
                        weights[i] + weights[j] - interactions[i][j][2]
                    )
                    antagonistic.append(
                        antagonistic_weight * min(c_i, c_j)
                        if interactions[i][j][1] == "min"
                        else antagonistic_weight * c_i * c_j
                    )

    return (
        sum(
            [
                weights[i]
                * concordance_marginal(
                    a[i],
                    b[i],
                    scales[i],
                    indifference_thresholds[i],
                    preference_thresholds[i],
                )
                for i in range(len(a))
            ]
        )
        + sum(mutual_strengthening)
        + sum(mutual_weakening)
        - sum(antagonistic)
    ) / (
        sum(weights)
        + sum(mutual_strengthening)
        + sum(mutual_weakening)
        - sum(antagonistic)
    )


def concordance_w_i(
    alternatives_perform: List[List[NumericValue]],
    scales: List[QuantitativeScale],
    weights: List[NumericValue],
    indifference_thresholds: List[Threshold],
    preference_thresholds: List[Threshold],
    interactions,  # missing typing
) -> List[List[NumericValue]]:
    return [
        [
            concordance_with_interactions(
                alternatives_perform[i],
                alternatives_perform[j],
                scales,
                weights,
                interactions,
                indifference_thresholds,
                preference_thresholds,
            )
            for j in range(len(alternatives_perform[i]))
        ]
        for i in range(len(alternatives_perform))
    ]
