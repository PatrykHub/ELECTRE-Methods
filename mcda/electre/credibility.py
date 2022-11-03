"""This module implements methods to compute
an outranking credibility."""

from functools import reduce
from typing import Dict, List, Union
import pandas as pd

from ..core.aliases import NumericValue
from ..core.scales import PreferenceDirection, QuantitativeScale


def credibility_cv_pair(
    concordance_comprehensive: NumericValue,
    discordance_comprehensive: NumericValue,
    counter_veto_occurs: List[Union[int, bool]],
) -> NumericValue:
    """_summary_

    :param concordance_comprehensive: _description_
    :param discordance_comprehensive: _description_
    :param counter_veto_occurs: _description_
    :return: _description_
    """
    return concordance_comprehensive * discordance_comprehensive ** (
        1 - sum(counter_veto_occurs) / len(counter_veto_occurs)
    )


def credibility_cv(
    concordance_comprehensive: pd.DataFrame,
    discordance_comprehensive: pd.DataFrame,
    counter_veto_occurs: pd.DataFrame,
) -> pd.DataFrame:
    """_summary_

    :param concordance_comprehensive: _description_
    :param discordance_comprehensive: _description_
    :param counter_veto_occurs: _description_
    :return: _description_
    """
    return pd.DataFrame(
        [
            [
                credibility_cv_pair(
                    concordance_comprehensive[alt_name_b][alt_name_a],
                    discordance_comprehensive[alt_name_b][alt_name_a],
                    [counter_veto_occurs[alt_name_b][alt_name_a]],
                )
                for alt_name_b in concordance_comprehensive.index
            ]
            for alt_name_a in concordance_comprehensive.index
        ],
        index=concordance_comprehensive.index,
        columns=concordance_comprehensive.index,
    )


def credibility_pair(
    concordance_comprehensive: NumericValue,
    discordance_comprehensive: NumericValue,
) -> NumericValue:
    """_summary_

    :param concordance_comprehensive: _description_
    :param discordance_comprehensive: _description_
    :return: _description_
    """
    return concordance_comprehensive * discordance_comprehensive


def credibility_comprehensive(
    concordance_comprehensive: pd.DataFrame,
    discordance_comprehensive: pd.DataFrame,
) -> pd.DataFrame:
    """_summary_

    :param concordance_comprehensive: _description_
    :param discordance_comprehensive: _description_
    :return: _description_
    """
    return pd.DataFrame(
        [
            [
                credibility_pair(
                    concordance_comprehensive[alt_name_b][alt_name_a],
                    discordance_comprehensive[alt_name_b][alt_name_a],
                )
                for alt_name_b in concordance_comprehensive.index
            ]
            for alt_name_a in concordance_comprehensive.index
        ],
        index=concordance_comprehensive.index,
        columns=concordance_comprehensive.index,
    )


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
    credibility: List[List[NumericValue]] = [
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
                    np_ba <= 1
                    and np_ab >= len(scales) / 2
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
    veto_threshold: List[NumericValue],
) -> List[List]:
    criteria_counts: List[List[List[float]]] = [
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
