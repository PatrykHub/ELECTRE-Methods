
from core.aliases import NumericValue
from typing import List, Union
from core.scales import QuantitativeScale, PreferenceDirection


def concordanceMarginal(a_value:               NumericValue,
                        b_value:               NumericValue,
                        scale:                 QuantitativeScale,
                        indifferenceThreshold: List[NumericValue],
                        preferenceThreshold:   List[NumericValue]) -> NumericValue:
    '''

    :param a_value:
    :param b_value:
    :param scale:
    :param indifferenceThreshold:
    :param preferenceThreshold:
    :return:
    '''
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
    '''

    :param a:
    :param b:
    :param scales:
    :param weights:
    :param indifferenceThreshold:
    :param preferenceThreshold:
    :return:
    '''
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


def check_interactions(interactions:    List[List[List]],
                       weights:         List[NumericValue],
                       length:          NumericValue) -> bool:
    '''

    :param interactions:
    :param weights:
    :param length:
    :return:
    '''

    if not isinstance(interactions, list) and not all(1 if isinstance(item, list) else 0 in interactions for item in interactions):
        raise TypeError('Interactions have to be represented as a matrix.')
    if not len(interactions) == length and not all(1 if len(row) == length else 0 in interactions for row in interactions):
        raise ValueError('Interactions have to be a square matrix.')
    for i in range(len(interactions)):
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
            if len(interactions[i][j]) == 3 and interactions[i][j][0] == 'MW' and weights[i] - abs(interactions[i][j][2]) < 0:
                raise ValueError('Incorrect interaction factor.')
            if len(interactions[i][j]) == 3 and interactions[i][j][0] == 'A' and weights[i] - interactions[i][j][2] < 0:
                raise ValueError('Incorrect interaction factor.')

    return True


def interact(interactions:      List[List[List]],
             g_i:               NumericValue,
             g_j:               NumericValue,
             interaction_token: str,
             z_function:        str,
             factor:            NumericValue):
    '''

    :param interactions:
    :param g_i:
    :param g_j:
    :param interaction_token:
    :param z_function:
    :param factor:
    :return:
    '''

    g1 = min(g_i, g_j) - 1
    g2 = max(g_i, g_j) - 1

    interactions[g1][g2].append(interaction_token)
    interactions[g1][g2].append(z_function)
    interactions[g1][g2].append(factor)



def concordanceWithInteractions(a:                     List[NumericValue],
                                b:                     List[NumericValue],
                                scales:                List[QuantitativeScale],
                                weights:               List[NumericValue],
                                interactions:          List[List[List]],
                                indifferenceThreshold: List[List[NumericValue]],
                                preferenceThreshold:   List[List[NumericValue]]) -> NumericValue:
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
    if len(a) != len(b) or len(a) != len(scales) or len(a) != len(weights) or\
            len(a) != len(indifferenceThreshold) or len(a) != len(preferenceThreshold):
        raise ValueError('All lists given in arguments have to have the same length.')
    for weight in weights:
        if not isinstance(weight, NumericValue):
            raise ValueError(f'Wrong weight type. Expected numeric value, got {type(weight).__name__}')

    check_interactions(interactions, weights, len(a))

    mutual_strengthening = []
    mutual_weakening = []
    antagonistic = []
    for i in range(len(interactions)):
        for j in range(i+1, len(interactions[i])):
            if len(interactions[i]) > 1:
                c_i = concordanceMarginal(a[i], b[i], scales[i], indifferenceThreshold[i], preferenceThreshold[i])
                c_j = concordanceMarginal(a[j], b[j], scales[j], indifferenceThreshold[j], preferenceThreshold[j])
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
                weights[i] * concordanceMarginal(a[i], b[i], scales[i], indifferenceThreshold[i], preferenceThreshold[i])
                for i in range(len(a))
            ]
            ) + sum(mutual_strengthening) + sum(mutual_weakening) - sum(antagonistic)) / \
           (sum(weights) + sum(mutual_strengthening) + sum(mutual_weakening) - sum(antagonistic))
