"""This module implements methods to compute concordance."""
from typing import List, Tuple, get_args

from ..core.aliases import NumericValue
from ..core.functions import Threshold
from ..core.scales import PreferenceDirection, QuantitativeScale
from ._validate import (_all_lens_equal, _both_values_in_scale,
                        _inverse_values, _weights_proper_vals)


def concordance_marginal(
    a_value: NumericValue,
    b_value: NumericValue,
    scale: QuantitativeScale,
    indifference_threshold: Threshold,
    preference_threshold: Threshold,
    inverse: bool = True,
) -> NumericValue:
    """
    :param a_value:
    :param b_value:
    :param scale:
    :param indifference_threshold:
    :param preference_threshold:
    :param inverse:
    :return:
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
    a_values: List[NumericValue],
    b_values: List[NumericValue],
    scales: List[QuantitativeScale],
    weights: List[NumericValue],
    indifference_thresholds: List[Threshold],
    preference_thresholds: List[Threshold],
    inverse: bool = False,
) -> NumericValue:
    """
    :param a:
    :param b:
    :param scales:
    :param weights:
    :param indifference_threshold:
    :param preference_threshold:
    :param inverse:
    :return:
    """
    _all_lens_equal(
        a_values=a_values,
        b_values=b_values,
        scales=scales,
        weights=weights,
        indifference_thresholds=indifference_thresholds,
        preference_thresholds=preference_thresholds,
    )
    _weights_proper_vals(weights)

    return sum(
        [
            weights[i]
            * concordance_marginal(
                a_values[i],
                b_values[i],
                scales[i],
                indifference_thresholds[i],
                preference_thresholds[i],
                inverse,
            )
            for i in range(len(a_values))
        ]
    ) / sum(weights)


def concordance_reinforced_pair(
    a_values: List[NumericValue],
    b_values: List[NumericValue],
    scales: List[QuantitativeScale],
    weights: List[NumericValue],
    indifference_thresholds: List[Threshold],
    preference_thresholds: List[Threshold],
    reinforced_thresholds: List[Threshold],
    reinforcement_factors: List[NumericValue],
    inverse: bool = False,
) -> NumericValue:
    """
    :param a:
    :param b:
    :param scales:
    :param weights:
    :param indifference_threshold:
    :param preference_threshold:
    :param reinforced_thresholds:
    :param reinforcement_factors:
    :param inverse:
    :return:
    """
    _all_lens_equal(
        a_values=a_values,
        b_values=b_values,
        scales=scales,
        weights=weights,
        indifference_thresholds=indifference_thresholds,
        preference_thresholds=preference_thresholds,
        reinforced_thresholds=reinforced_thresholds,
        reinforcement_factors=reinforcement_factors,
    )

    # TODO: check if reinforced threshold are > preference
    reinforced_threshold_vals: List[NumericValue] = [
        reinforced_threshold(a_value)
        for reinforced_threshold, a_value in zip(reinforced_thresholds, a_values)
    ]

    reinforce_occur = [
        a_values[i] - b_values[i] > reinforced_threshold_vals[i]
        if scales[i].preference_direction == PreferenceDirection.MIN
        else a_values[i] - b_values[i] < reinforced_threshold_vals[i]
        for i in range(len(a_values))
    ]

    sum_weights_thresholds = sum(
        [
            weights[i] * reinforce_occur[i] * reinforcement_factors[i]
            for i in range(len(reinforcement_factors))
        ]
    )

    return (
        sum_weights_thresholds
        + sum(
            [
                0
                if reinforce_occur[i]
                else weights[i]
                * concordance_marginal(
                    a_values[i],
                    b_values[i],
                    scales[i],
                    indifference_thresholds[i],
                    preference_thresholds[i],
                    inverse,
                )
                for i in range(len(a_values))
            ]
        )
    ) / (
        sum_weights_thresholds
        + sum([weights[i] * (not reinforce_occur[i]) for i in range(len(weights))])
    )


def concordance_reinforced(
    alternatives_perform: List[List[NumericValue]],
    scales: List[QuantitativeScale],
    weights: List[NumericValue],
    indifference_threshold: List[Threshold],
    preference_threshold: List[Threshold],
    reinforced_thresholds: List[Threshold],
    reinforcement_factors: List[NumericValue],
    profiles_perform: List[List[NumericValue]] = None,
):
    if profiles_perform is not None:
        return [
            [
                concordance_reinforced_pair(
                    alternatives_perform[i],
                    profiles_perform[j],
                    scales,
                    weights,
                    indifference_threshold,
                    preference_threshold,
                    reinforced_thresholds,
                    reinforcement_factors,
                    True,
                )
                for j in range(len(profiles_perform))
            ]
            for i in range(len(alternatives_perform))
        ]

    return [
        [
            concordance_reinforced_pair(
                alternatives_perform[i],
                alternatives_perform[j],
                scales,
                weights,
                indifference_threshold,
                preference_threshold,
                reinforced_thresholds,
                reinforcement_factors,
            )
            for j in range(len(alternatives_perform[i]))
        ]
        for i in range(len(alternatives_perform))
    ]


def concordance(
    alternatives_perform: List[List[NumericValue]],
    scales: List[QuantitativeScale],
    weights: List[NumericValue],
    indifference_threshold: List[Threshold],
    preference_threshold: List[Threshold],
    profiles_perform: List[List[NumericValue]] = None,
) -> List[List[NumericValue]]:
    if profiles_perform is not None:
        return [
            [
                concordance_comprehensive(
                    alternatives_perform[i],
                    profiles_perform[j],
                    scales,
                    weights,
                    indifference_threshold,
                    preference_threshold,
                    True,
                )
                for j in range(len(profiles_perform))
            ]
            for i in range(len(alternatives_perform))
        ]

    return [
        [
            concordance_comprehensive(
                alternatives_perform[i],
                alternatives_perform[j],
                scales,
                weights,
                indifference_threshold,
                preference_threshold,
            )
            for j in range(len(alternatives_perform[i]))
        ]
        for i in range(len(alternatives_perform))
    ]


def concordance_profiles_thresholds(
    alternatives_perform: List[List[NumericValue]],
    scales: List[QuantitativeScale],
    weights: List[NumericValue],
    indifference_threshold: List[List[Threshold]],
    preference_threshold: List[List[Threshold]],
    profiles_perform: List[List[NumericValue]],
) -> Tuple[List[List[NumericValue]], List[List[NumericValue]]]:
    """
    :param alternatives_perform:
    :param scales:
    :param weights:
    :param indifference_threshold:
    :param preference_threshold:
    :param profiles_perform:
    :return:
    """
    return [
        [
            concordance_comprehensive(
                alternatives_perform[i],
                profiles_perform[j],
                scales,
                weights,
                indifference_threshold[j],
                preference_threshold[j],
                True,
            )
            for j in range(len(profiles_perform))
        ]
        for i in range(len(alternatives_perform))
    ], [
        [
            concordance_comprehensive(
                profiles_perform[j],
                alternatives_perform[i],
                scales,
                weights,
                indifference_threshold[j],
                preference_threshold[j],
            )
            for j in range(len(profiles_perform))
        ]
        for i in range(len(alternatives_perform))
    ]


# TODO: change interaction type to enum
# TODO: interaction as object instance maybeeee ???
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
