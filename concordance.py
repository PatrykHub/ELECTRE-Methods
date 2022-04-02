
from core.aliases import NumericValue
from typing import List
from core.scales import QuantitativeScale, PreferenceDirection


def concordanceMarginal(a_value:               NumericValue,
                        b_value:               NumericValue,
                        scale:                 QuantitativeScale,
                        indifferenceThreshold: List[NumericValue],
                        preferenceThreshold:   List[NumericValue]) -> NumericValue:
    """

    :param a_value:
    :param b_value:
    :param scale:
    :param indifferenceThreshold:
    :param preferenceThreshold:
    :return:
    """
    if not isinstance(a_value, NumericValue) or not isinstance(b_value, NumericValue):
        raise TypeError('Both criteria values have to be numeric values (int or float).')

    if not isinstance(scale, QuantitativeScale):
        raise TypeError(f'Wrong scale type. Expected QuantitativeScale, got {type(scale).__name__} instead.')

    if not isinstance(indifferenceThreshold, list) or\
            not isinstance(preferenceThreshold, list) or\
            len(indifferenceThreshold) != 2 or len(preferenceThreshold) != 2:
        raise TypeError('Both thresholds have to be a two elements list.')

    if a_value not in scale or b_value not in scale:
        raise ValueError('Both criteria values have to be between the min and max of the given interval.')

    try:
        q_a: NumericValue = indifferenceThreshold[0] * a_value + indifferenceThreshold[1]
        p_a: NumericValue =   preferenceThreshold[0] * a_value +   preferenceThreshold[1]

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


def concordanceComprehensive(a:                     List[NumericValue],
                             b:                     List[NumericValue],
                             scales:                List[QuantitativeScale],
                             weights:               List[NumericValue],
                             indifferenceThreshold: List[List[NumericValue]],
                             preferenceThreshold:   List[List[NumericValue]]) -> NumericValue:
    """

    :param a:
    :param b:
    :param scales:
    :param weights:
    :param indifferenceThreshold:
    :param preferenceThreshold:
    :return:
    """
    if not (isinstance(a, list) and isinstance(b, list) and isinstance(scales, list) and isinstance(weights, list) and
            isinstance(indifferenceThreshold, list) and isinstance(preferenceThreshold, list)):
        raise TypeError('All arguments have to be lists.')
    if len(a) != len(b) or len(a) != len(scales) or len(a) != len(weights) or\
            len(a) != len(indifferenceThreshold) or len(a) != len(preferenceThreshold):
        raise ValueError('All lists given in arguments have to have the same length.')
    for weight in weights:
        if not isinstance(weight, NumericValue):
            raise ValueError(f'Wrong weight type. Expected numeric value, got {type(weight).__name__}')
    return sum(
            [
                weights[i] * concordanceMarginal(a[i], b[i], scales[i], indifferenceThreshold[i], preferenceThreshold[i])
                for i in range(len(a))
            ]
        ) / sum(weights)


def concordance_reinforced_pair(a:                      List[NumericValue],
                                b:                      List[NumericValue],
                                scales:                 List[QuantitativeScale],
                                weights:                List[NumericValue],
                                indifferenceThreshold:  List[List[NumericValue]],
                                preferenceThreshold:    List[List[NumericValue]],
                                reinforcedThresholds:   List[List[NumericValue]],
                                reinforcementFactors:   List[NumericValue]) -> NumericValue:
    """

    :param a:
    :param b:
    :param scales:
    :param weights:
    :param indifferenceThreshold:
    :param preferenceThreshold:
    :param reinforcedThresholds:
    :param reinforcementFactors:
    :return:
    """

    try:
        reinforcedThresholds = [
            (reinforcedThresholds[i][0] * a[i] + reinforcedThresholds[i][1]) for i in range(len(reinforcedThresholds))
        ]
    except TypeError as e:
        e.args = ('Threshold values have to be numeric values (int or float).',)
        raise

    reinforce_occur = [
        a[i] - b[i] > reinforcedThresholds[i] if scales[i].preference_direction == PreferenceDirection.MIN else
        a[i] - b[i] < reinforcedThresholds[i]
        for i in range(len(a))
    ]

    sum_weights_thresholds = sum([
        weights[i] * reinforce_occur[i] * reinforcementFactors[i] for i in range(len(reinforcementFactors))
    ])

    return (
        sum_weights_thresholds + sum([
            0 if reinforce_occur[i] else
            weights[i] * concordanceMarginal(a[i], b[i], scales[i], indifferenceThreshold[i], preferenceThreshold[i])
            for i in range(len(a))
        ])
    ) / (sum_weights_thresholds + sum([weights[i] * (not reinforce_occur[i]) for i in range(len(weights))]))


def concordance_reinforced(alternativesPerform:   List[List[NumericValue]],
                           scales:                List[QuantitativeScale],
                           weights:               List[NumericValue],
                           indifferenceThreshold: List[List[NumericValue]],
                           preferenceThreshold:   List[List[NumericValue]],
                           reinforcedThresholds:  List[List[NumericValue]],
                           reinforcementFactors:  List[NumericValue],
                           profilesPerform:       List[List[NumericValue]] = None) -> List[List[NumericValue]]:
    if profilesPerform is not None:
        return [
            [concordance_reinforced_pair(
                alternativesPerform[i],
                profilesPerform[j],
                scales, weights,
                indifferenceThreshold, preferenceThreshold,
                reinforcedThresholds, reinforcementFactors
            )
                for j in range( len(profilesPerform) )
            ]
            for i in range(len(alternativesPerform))
        ]

    return [
        [concordance_reinforced_pair(
            alternativesPerform[i],
            alternativesPerform[j],
            scales, weights,
            indifferenceThreshold, preferenceThreshold,
            reinforcedThresholds, reinforcementFactors
        )
            for j in range( len(alternativesPerform[i]) )
        ]
        for i in range(len(alternativesPerform))
    ]


def concordance(alternativesPerform:   List[List[NumericValue]],
                scales:                List[QuantitativeScale],
                weights:               List[NumericValue],
                indifferenceThreshold: List[List[NumericValue]],
                preferenceThreshold:   List[List[NumericValue]],
                profilesPerform:       List[List[NumericValue]] = None) -> List[List[NumericValue]]:
    if profilesPerform is not None:
        return [
            [concordanceComprehensive(
                alternativesPerform[i],
                profilesPerform[j],
                scales, weights,
                indifferenceThreshold, preferenceThreshold
            )
                for j in range(len(profilesPerform))
            ]
            for i in range(len(alternativesPerform))
        ]

    return [
        [concordanceComprehensive(
            alternativesPerform[i],
            alternativesPerform[j],
            scales, weights,
            indifferenceThreshold, preferenceThreshold
        )
            for j in range(len(alternativesPerform[i]))
        ]
        for i in range(len(alternativesPerform))
    ]

