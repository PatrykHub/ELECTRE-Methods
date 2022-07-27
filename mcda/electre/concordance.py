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

def concordance_with_interactions(a: List[NumericValue],
                                  b: List[NumericValue],
                                  scales: List[QuantitativeScale],
                                  weights: List[NumericValue],
                                  interactions: List[List[List]],
                                  indifferenceThreshold: List[List[NumericValue]],
                                  preferenceThreshold: List[List[NumericValue]]) -> NumericValue:
    '''
    :param a:
    :param b:
    :param scales:
    :param weights:
    :param interactions:
    :param indifferenceThreshold:
    :param preferenceThreshold:
    :return:
    '''
    if not (isinstance(a, list) and isinstance(b, list) and isinstance(scales, list) and isinstance(weights, list) and
            isinstance(indifferenceThreshold, list) and isinstance(preferenceThreshold, list)):
        raise TypeError('All arguments have to be lists.')
    if len(a) != len(b) or len(a) != len(scales) or len(a) != len(weights) or \
            len(a) != len(indifferenceThreshold) or len(a) != len(preferenceThreshold):
        raise ValueError('All lists given in arguments have to have the same length.')
    for weight in weights:
        if not isinstance(weight, NumericValue):
            raise ValueError(f'Wrong weight type. Expected numeric value, got {type(weight).__name__}')
    if not isinstance(interactions, list) and not all(
            1 if isinstance(item, list) else 0 in interactions for item in interactions):
        raise TypeError('Interactions have to be represented as a matrix.')
    if not len(interactions) == len(a) and not all(
            1 if len(row) == len(a) else 0 in interactions for row in interactions):
        raise ValueError('Interactions have to be a square matrix.')
    for i in range(len(interactions)):
        if interactions[i][i] != []:
            raise ValueError('Criterion cannot interact with itself.')
        for j in range(len(interactions[i])):
            if len(interactions[i][j]) > 0 and len(interactions[i][j]) != 3:
                raise ValueError('Each interaction has to be represented as a list of length 3.')
            if len(interactions[i][j]) == 3 and interactions[i][j][0] not in ['MW', 'MS', 'A']:
                raise ValueError('The interaction type has to be represented by one of the following tokens:\n\'MW\' '
                                 '- Mutual Weakening\n\'MS\' - Mutual Strengthening\n\'A\' - Antagonistic')
            if len(interactions[i][j]) == 3 and interactions[i][j][1] not in ['min', 'multi']:
                raise ValueError('The Z function has to be represented by one of the following tokens:\n\'min\' - '
                                 'minimum\n\'multi\' - multiplication')
            if len(interactions[i][j]) == 3 and not isinstance(interactions[i][j][2], NumericValue):
                raise TypeError('Interaction factor must be a numerical value.')
            if len(interactions[i][j]) == 3 and interactions[i][j][0] == 'MW' and weights[i] - abs(
                    interactions[i][j][2]) < 0:
                raise ValueError('Incorrect interaction factor.')
            if len(interactions[i][j]) == 3 and interactions[i][j][0] == 'A' and weights[i] - interactions[i][j][2] < 0:
                raise ValueError('Incorrect interaction factor.')

    mutual_strengthening = []
    mutual_weakening = []
    antagonistic = []
    for i in range(len(interactions)):
        for j in range(len(interactions[i])):
            if len(interactions[i][j]) > 1:
                c_i = concordance_marginal(a[i], b[i], scales[i], indifferenceThreshold[i], preferenceThreshold[i])
                c_j = concordance_marginal(a[j], b[j], scales[j], indifferenceThreshold[j], preferenceThreshold[j])
                if interactions[i][j][0] == 'MS':
                    strengthening_weight = weights[i] + weights[j] + interactions[i][j][2]
                    mutual_strengthening.append(strengthening_weight * min(c_i, c_j) if interactions[i][j][1] == 'min'
                                                else strengthening_weight * c_i * c_j)
                elif interactions[i][j][0] == 'MW':
                    weakening_weight = weights[i] + weights[j] + interactions[i][j][2]
                    mutual_weakening.append(weakening_weight * min(c_i, c_j) if interactions[i][j][1] == 'min'
                                            else weakening_weight * c_i * c_j)
                else:
                    antagonistic_weight = weights[i] + weights[j] - interactions[i][j][2]
                    antagonistic.append(antagonistic_weight * min(c_i, c_j) if interactions[i][j][1] == 'min'
                                        else antagonistic_weight * c_i * c_j)

    return (sum(
        [
            weights[i] * concordance_marginal(a[i], b[i], scales[i], indifferenceThreshold[i], preferenceThreshold[i])
            for i in range(len(a))
        ]
    ) + sum(mutual_strengthening) + sum(mutual_weakening) - sum(antagonistic)) / \
           (sum(weights) + sum(mutual_strengthening) + sum(mutual_weakening) - sum(antagonistic))


def concordance_w_i(alternatives_perform: List[List[NumericValue]],
                    scales: List[QuantitativeScale],
                    weights: List[NumericValue],
                    indifference_threshold: List[List[NumericValue]],
                    preference_threshold: List[List[NumericValue]],
                    interactions
                    ):
    return [
        [concordance_with_interactions(
            alternatives_perform[i],
            alternatives_perform[j],
            scales, weights,
            interactions,
            indifference_threshold, preference_threshold
        )
            for j in range(len(alternatives_perform[i]))
        ]
        for i in range(len(alternatives_perform))
    ]