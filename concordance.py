from core.aliases import NumericValue
from typing import List
from core.scales import QuantitativeScale, PreferenceDirection


def concordance_marginal(a_value: NumericValue,
                         b_value: NumericValue,
                         scale: QuantitativeScale,
                         indifference_threshold: List[NumericValue],
                         preference_threshold: List[NumericValue],
                         inverse: bool) -> NumericValue:
    '''
    :param a_value:
    :param b_value:
    :param scale:
    :param indifference_threshold:
    :param preference_threshold:
    :param inverse:
    :return:
    '''
    if not isinstance(a_value, NumericValue) or not isinstance(b_value, NumericValue):
        raise TypeError('Both criteria values have to be numeric values (int or float).')

    if not isinstance(scale, QuantitativeScale):
        raise TypeError(f'Wrong scale type. Expected QuantitativeScale, got {type(scale).__name__} instead.')

    if not isinstance(indifference_threshold, list) or \
            not isinstance(preference_threshold, list) or \
            len(indifference_threshold) != 2 or len(preference_threshold) != 2:
        raise TypeError('Both thresholds have to be a two elements list.')

    if a_value not in scale or b_value not in scale:
        raise ValueError('Both criteria values have to be between the min and max of the given interval.')

    try:
        if inverse:
            a_value, b_value = b_value, a_value
            if scale.preference_direction == PreferenceDirection.MIN:
                scale.preference_direction = PreferenceDirection.MAX
            else:
                scale.preference_direction = PreferenceDirection.MIN

        q_a: NumericValue = indifference_threshold[0] * a_value + indifference_threshold[1]
        p_a: NumericValue = preference_threshold[0] * a_value + preference_threshold[1]

        if q_a >= p_a:
            raise ValueError('Indifference threshold can\'t be bigger than the preference threshold.')

        if scale.preference_direction == PreferenceDirection.MIN:
            if a_value - b_value <= q_a:
                return 1
            elif a_value - b_value > p_a:
                return 0
            return (b_value - a_value + p_a) / (p_a - q_a)

        if b_value - a_value <= q_a:
            return 1
        elif b_value - a_value > p_a:
            return 0
        return (p_a - (b_value - a_value)) / (p_a - q_a)

    except TypeError as e:
        e.args = ('Threshold values have to be numeric values (int or float).',)
        raise


def concordance_comprehensive(a: List[NumericValue],
                              b: List[NumericValue],
                              scales: List[QuantitativeScale],
                              weights: List[NumericValue],
                              indifference_threshold: List[List[NumericValue]],
                              preference_threshold: List[List[NumericValue]],
                              inverse: bool = False) -> NumericValue:
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
    if not (isinstance(a, list) and isinstance(b, list) and isinstance(scales, list) and isinstance(weights, list) and
            isinstance(indifference_threshold, list) and isinstance(preference_threshold, list)):
        raise TypeError('All arguments have to be lists.')
    if len(a) != len(b) or len(a) != len(scales) or len(a) != len(weights) or \
            len(a) != len(indifference_threshold) or len(a) != len(preference_threshold):
        raise ValueError('All lists given in arguments have to have the same length.')
    for weight in weights:
        if not isinstance(weight, NumericValue):
            raise ValueError(f'Wrong weight type. Expected numeric value, got {type(weight).__name__}')
    return sum(
        [
            weights[i] * concordance_marginal(a[i], b[i], scales[i], indifference_threshold[i], preference_threshold[i],
                                              inverse)
            for i in range(len(a))
        ]
    ) / sum(weights)


def concordance_reinforced_pair(a: List[NumericValue],
                                b: List[NumericValue],
                                scales: List[QuantitativeScale],
                                weights: List[NumericValue],
                                indifference_threshold: List[List[NumericValue]],
                                preference_threshold: List[List[NumericValue]],
                                reinforced_thresholds: List[List[NumericValue]],
                                reinforcement_factors: List[NumericValue],
                                inverse: bool = False) -> NumericValue:
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

    try:
        reinforced_thresholds = [
            (reinforced_thresholds[i][0] * a[i] + reinforced_thresholds[i][1]) for i in
            range(len(reinforced_thresholds))
        ]
    except TypeError as e:
        e.args = ('Threshold values have to be numeric values (int or float).',)
        raise

    reinforce_occur = [
        a[i] - b[i] > reinforced_thresholds[i] if scales[i].preference_direction == PreferenceDirection.MIN else
        a[i] - b[i] < reinforced_thresholds[i]
        for i in range(len(a))
    ]

    sum_weights_thresholds = sum([
        weights[i] * reinforce_occur[i] * reinforcement_factors[i] for i in range(len(reinforcement_factors))
    ])

    return (
                   sum_weights_thresholds + sum([
               0 if reinforce_occur[i] else
               weights[i] * concordance_marginal(a[i], b[i], scales[i], indifference_threshold[i],
                                                 preference_threshold[i], inverse)
               for i in range(len(a))
           ])
           ) / (sum_weights_thresholds + sum([weights[i] * (not reinforce_occur[i]) for i in range(len(weights))]))


def concordance_reinforced(alternatives_perform: List[List[NumericValue]],
                           scales: List[QuantitativeScale],
                           weights: List[NumericValue],
                           indifference_threshold: List[List[NumericValue]],
                           preference_threshold: List[List[NumericValue]],
                           reinforced_thresholds: List[List[NumericValue]],
                           reinforcement_factors: List[NumericValue],
                           profiles_perform: List[List[NumericValue]] = None):
    if profiles_perform is not None:
        return [
            [concordance_reinforced_pair(
                alternatives_perform[i],
                profiles_perform[j],
                scales, weights,
                indifference_threshold, preference_threshold,
                reinforced_thresholds, reinforcement_factors, True
            )
                for j in range(len(profiles_perform))
            ]
            for i in range(len(alternatives_perform))
        ]

    return [
        [concordance_reinforced_pair(
            alternatives_perform[i],
            alternatives_perform[j],
            scales, weights,
            indifference_threshold, preference_threshold,
            reinforced_thresholds, reinforcement_factors
        )
            for j in range(len(alternatives_perform[i]))
        ]
        for i in range(len(alternatives_perform))
    ]


def concordance(alternatives_perform: List[List[NumericValue]],
                scales: List[QuantitativeScale],
                weights: List[NumericValue],
                indifference_threshold: List[List[NumericValue]],
                preference_threshold: List[List[NumericValue]],
                profiles_perform: List[List[NumericValue]] = None):
    if profiles_perform is not None:
        return [
            [concordance_comprehensive(
                alternatives_perform[i],
                profiles_perform[j],
                scales, weights,
                indifference_threshold, preference_threshold, True
            )
                for j in range(len(profiles_perform))
            ]
            for i in range(len(alternatives_perform))
        ]

    return [
        [concordance_comprehensive(
            alternatives_perform[i],
            alternatives_perform[j],
            scales, weights,
            indifference_threshold, preference_threshold
        )
            for j in range(len(alternatives_perform[i]))
        ]
        for i in range(len(alternatives_perform))
    ]



def concordance_profiles_thresholds(alternatives_perform: List[List[NumericValue]],
                                    scales: List[QuantitativeScale],
                                    weights: List[NumericValue],
                                    indifference_threshold: List[List[List[NumericValue]]],
                                    preference_threshold: List[List[List[NumericValue]]],
                                    profiles_perform: List[List[NumericValue]]):
    '''

    :param alternatives_perform:
    :param scales:
    :param weights:
    :param indifference_threshold:
    :param preference_threshold:
    :param profiles_perform:
    :return:
    '''
    return [
        [concordance_comprehensive(
            alternatives_perform[i],
            profiles_perform[j],
            scales, weights,
            indifference_threshold[j], preference_threshold[j], True
        )
            for j in range(len(profiles_perform))
        ]
        for i in range(len(alternatives_perform))
    ], [
        [concordance_comprehensive(
            profiles_perform[j],
            alternatives_perform[i],
            scales, weights,
            indifference_threshold[j], preference_threshold[j]
        )
            for j in range(len(profiles_perform))
        ]
        for i in range(len(alternatives_perform))
    ]
