from core.aliases import NumericValue
from typing import List
from functools import reduce


def credibility_marginal(concordance: NumericValue,
                         discordance: NumericValue):
    '''
    :param x:
    :param y:
    :param concordance:
    :param discordance:
    :return:
    '''
    discordance_values = [discordance]
    if set(discordance_values) == {0}:  # only zeros
        c_idx = concordance
    elif 1 in discordance_values:  # at least one '1'
        c_idx = 0.0
    else:
        factors = []
        for d in discordance_values:
            factor = None
            if d > concordance:
                factor = (1 - d) / (1 - concordance)
            if factor:
                factors.append(factor)
        if factors == []:
            c_idx = concordance
        else:
            c_idx = concordance * reduce(lambda f1, f2: f1 * f2, factors)
    return c_idx


def credibility_comprehensive(comparables_a: List[NumericValue],
                              comparables_b: List[NumericValue],
                              concordance: List[List[NumericValue]],
                              discordance: List[List[NumericValue]]):

    credibility = {}

    for i in comparables_a:
        for j in comparables_b:
            credibility[i] = {}
            credibility[i].update({j: credibility_marginal(concordance[i][j], discordance[i][j])})
            credibility[j] = {}
            credibility[j].update({i: credibility_marginal(concordance[j][i], discordance[j][i])})

    return credibility

