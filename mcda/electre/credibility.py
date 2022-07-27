"""This module implements methods to compute
an outranking credibility."""

from typing import List, Union
from functools import reduce
from ..core.aliases import NumericValue


"""This module implements methods to compute
an outranking credibility."""

from typing import List, Union

from ..core.aliases import NumericValue


def credibility_cv_pair(
    concordance_comprehensive: NumericValue,
    discordance_comprehensive: NumericValue,
    counter_veto_occurs: List[Union[int, bool]],
) -> NumericValue:
    """_summary_
    :param NumericValue concordance_comprehensive: _description_
    :param NumericValue discordance_comprehensive: _description_
    :param List[Union[int, bool]] counter_veto_occurs: _description_
    :return NumericValue: _description_
    """
    try:
        return concordance_comprehensive * discordance_comprehensive ** (
            1 - sum(counter_veto_occurs) / len(counter_veto_occurs)
        )
    except TypeError as exc:
        exc.args = ("",)
        raise  # TODO


def credibility_cv(
    concordance_comprehensive: List[List[NumericValue]],
    discordance_comprehensive: List[List[NumericValue]],
    counter_veto_occurs: List[List[int]],
) -> List[List[NumericValue]]:
    """_summary_
    :param List[List[NumericValue]] concordance_comprehensive: _description_
    :param List[List[NumericValue]] discordance_comprehensive: _description_
    :param List[List[int]] counter_veto_occurs: _description_
    :return List[List[NumericValue]]: _description_
    """
    try:
        return [
            [
                credibility_cv_pair(concordance, discordance, [cv])
                for concordance, discordance, cv in zip(
                    concordance_row, discordance_row, cv_row
                )
            ]
            for concordance_row, discordance_row, cv_row in zip(
                concordance_comprehensive,
                discordance_comprehensive,
                counter_veto_occurs,
            )
        ]
    except TypeError as exc:
        exc.args = ("",)
        raise  # TODO

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

from typing import List
from mcda.core.aliases import NumericValue
from mcda.core.scales import QuantitativeScale, PreferenceDirection


def _get_criteria_difference(
    a_value: NumericValue, b_value: NumericValue, scale: QuantitativeScale
) -> NumericValue:
    return (
        a_value - b_value
        if scale.preference_direction == PreferenceDirection.MAX
        else b_value - a_value
    )


def _get_criteria_counts(
    a: List[NumericValue],
    b: List[NumericValue],
    scales: List[QuantitativeScale],
    indifference_threshold: List[NumericValue],
    prefference_threshold: List[NumericValue],
) -> List:
    np = nq = ni = no = 0

    for i in range(len(a)):
        difference = _get_criteria_difference(a[i], b[i], scales[i])

        if difference:
            if difference >= prefference_threshold[i]:
                np += 1
            elif difference > indifference_threshold[i]:
                nq += 1
            else:
                ni += 1
        else:
            no += 1

    return [np, nq, ni, no]


def _is_veto(
    a: List[NumericValue],
    b: List[NumericValue],
    scales: List[QuantitativeScale],
    veto_threshold: List[NumericValue],
) -> bool:
    for i in range(len(a)):
        if (
            veto_threshold[i] is not None
            and _get_criteria_difference(a[i], b[i], scales[i]) > veto_threshold[i]
        ):
            return True
    return False


def _get_credibility_values(
    alternatives: List[List],
    criteria_counts: List[List],
    scales: List[QuantitativeScale],
    veto_threshold: List[NumericValue],
) -> List[List]:
    credibility = [
        [0 for _ in range(len(alternatives))] for _ in range(len(alternatives))
    ]

    for i in range(len(alternatives)):
        for j in range(len(alternatives)):
            if i == j:
                credibility[i][j] = 1.0
            else:
                np_ab, nq_ab, ni_ab = criteria_counts[i][j][0:3]
                np_ba, nq_ba, ni_ba = criteria_counts[j][i][0:3]

                if np_ba + nq_ba == 0 and ni_ba < np_ab + nq_ab + ni_ab:
                    credibility[i][j] = 1.0

                elif np_ba == 0:
                    if nq_ba <= np_ab and nq_ba + ni_ba < np_ab + nq_ab + ni_ab:
                        credibility[i][j] = 0.8

                    elif nq_ba <= np_ab + nq_ab:
                        credibility[i][j] = 0.6

                    else:
                        credibility[i][j] = 0.4

                elif (
                    np_ba <= 1 and np_ab >= len(scales) / 2
                    and _is_veto(
                        alternatives[j], alternatives[i], scales, veto_threshold
                    )
                    is False
                ):
                    credibility[i][j] = 0.2

    return credibility


def credibility_electre_iv(
    alternatives: List[List],
    scales: List[QuantitativeScale],
    indifference_threshold: List[NumericValue],
    prefference_threshold: List[NumericValue],
    veto_threshold: List[NumericValue] = None,
) -> List[List]:
    criteria_counts = [
        [[0.0] * 4 for _ in range(len(alternatives))] for _ in range(len(alternatives))
    ]

    for i in range(len(alternatives)):
        for j in range(len(alternatives)):
            if i != j:
                criteria_counts[i][j] = _get_criteria_counts(
                    alternatives[i],
                    alternatives[j],
                    scales,
                    indifference_threshold,
                    prefference_threshold,
                )

    return _get_credibility_values(
        alternatives, criteria_counts, scales, veto_threshold
    )