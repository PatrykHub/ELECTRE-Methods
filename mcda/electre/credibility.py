"""This module implements methods to compute
an outranking credibility."""

from typing import List, Union
import pandas as pd
import math
from functools import reduce


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


def get_criteria_counts(
    a_values: List[NumericValue],
    b_values: List[NumericValue],
    scales: List[QuantitativeScale],
    indifference_thresholds: List[NumericValue],
    preference_thresholds: List[NumericValue],
) -> List[int]:
    np = nq = ni = no = 0

    for i in range(len(a_values)):
        difference = _get_criteria_difference(a_values[i], b_values[i], scales[i])

        if difference > 0.0:
            if difference >= preference_thresholds[i]:
                np += 1
            elif difference > indifference_thresholds[i]:
                nq += 1
            else:
                ni += 1
        elif math.isclose(difference, 0.0):
            no += 1

    return [np, nq, ni, no]


def _is_veto(
    a_values: List[NumericValue],
    b_values: List[NumericValue],
    scales: List[QuantitativeScale],
    veto_thresholds: List[NumericValue],
) -> bool:
    for i in range(len(a_values)):
        if (
            veto_thresholds[i] is not None
            and _get_criteria_difference(a_values[i], b_values[i], scales[i]) > veto_thresholds[i]
        ):
            return True
    return False


def get_credibility_values(
    alternatives: List[List[NumericValue]],
    criteria_counts: List[List[List[int]]],
    scales: List[QuantitativeScale],
    veto_thresholds: List[NumericValue],
) -> List[List[NumericValue]]:
    credibility: List[List[NumericValue]] = [
        [0.0 for _ in range(len(alternatives))] for _ in range(len(alternatives))
    ]

    for i in range(len(alternatives)):
        for j in range(len(alternatives)):
            if i == j:
                credibility[i][j] = 1.0
            else:
                np_ab, nq_ab, ni_ab = criteria_counts[i][j][:3]
                np_ba, nq_ba, ni_ba = criteria_counts[j][i][:3]

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
                        alternatives[j], alternatives[i], scales, veto_thresholds
                    )
                    is False
                ):
                    credibility[i][j] = 0.2

    return credibility


def credibility_electre_iv(
    alternatives: List[List[NumericValue]],
    scales: List[QuantitativeScale],
    indifference_thresholds: List[NumericValue],
    preference_thresholds: List[NumericValue],
    veto_thresholds: List[NumericValue],
) -> List[List[NumericValue]]:
    criteria_counts: List[List[List[int]]] = [
        [[0] * 4 for _ in range(len(alternatives))] for _ in range(len(alternatives))
    ]

    for i in range(len(alternatives)):
        for j in range(len(alternatives)):
            if i != j:
                criteria_counts[i][j] = get_criteria_counts(
                    alternatives[i],
                    alternatives[j],
                    scales,
                    indifference_thresholds,
                    preference_thresholds,
                )

    return get_credibility_values(
        alternatives, criteria_counts, scales, veto_thresholds
    )
