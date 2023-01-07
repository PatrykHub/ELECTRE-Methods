"""This module implements methods to compute an outranking credibility."""
import math
from enum import Enum
from typing import Optional, Sized, Tuple, Union

import numpy as np
import pandas as pd

from ..core.aliases import NumericValue
from .utils import get_criterion_difference, is_veto


def credibility_pair(
    concordance_comprehensive: NumericValue,
    discordance: NumericValue,
) -> NumericValue:
    """Computes the credibility value S(a, b) of an outranking relation, based on
    comprehensive concordance C(a, b) and discordance Delta_CD(a,b)
    or non-discordance Delta(a, b) indices.

    :param concordance_comprehensive: comprehensive concordance value C(a, b)
    :param discordance: discordance Delta_CD(a,b) or non-discordance value Delta(a, b)


    :return: Credibility value S(a, b), value from [0, 1] interval
    """
    return concordance_comprehensive * discordance


def credibility_comprehensive(
    concordance_comprehensive: pd.DataFrame,
    discordance: pd.DataFrame,
    is_non_discordance: bool = True,
) -> pd.DataFrame:
    """Computes the credibility of an outranking relation, based on
    comprehensive concordance matrix and discordance or non-discordance matrix.

    :param concordance_comprehensive: comprehensive concordance matrix
    :param discordance: discordance matrix
    :param is_non_discordance: indicates if in input is discordance or non-discordance matrix

    :return: Credibility matrix with float values from [0, 1] interval
    """
    if not is_non_discordance:
        discordance = 1 - discordance
    return pd.DataFrame(
        [
            [
                credibility_pair(
                    concordance_comprehensive[alt_name_b][alt_name_a],
                    discordance[alt_name_b][alt_name_a],
                )
                for alt_name_b in concordance_comprehensive.columns.values
            ]
            for alt_name_a in concordance_comprehensive.index.values
        ],
        index=concordance_comprehensive.index,
        columns=concordance_comprehensive.columns,
    )


def credibility_cv_pair(
    concordance_comprehensive: NumericValue,
    non_discordance: NumericValue,
    counter_veto_occurs: Union[int, Sized],
    number_of_criteria: int,
) -> NumericValue:
    """Computes the credibility value S(a, b) of an outranking relation, based on
    comprehensive concordance C(a, b) and non-discordance Delta(a, b) indices,
    also counter veto occurs cv(a, b) and number of criteria.

    :param concordance_comprehensive: comprehensive concordance value C(a, b)
    :param non_discordance: non-discordance value Delta(a, b)
    :param counter_veto_occurs: counter veto occurs cv(a, b)
    :param number_of_criteria: number of criteria

    :return: Credibility value S(a, b), value from [0, 1] interval
    """
    return concordance_comprehensive * non_discordance ** (
        1
        - (
            len(counter_veto_occurs)
            if isinstance(counter_veto_occurs, Sized)
            else counter_veto_occurs
        )
        / number_of_criteria
    )


def credibility_cv(
    concordance_comprehensive: pd.DataFrame,
    non_discordance: pd.DataFrame,
    counter_veto_occurs: pd.DataFrame,
    number_of_criteria: int,
) -> pd.DataFrame:
    """Computes the credibility of an outranking relation, based on
    comprehensive concordance matrix, non_discordance matrix,
    counter veto occurs matrix and number of criteria.

    :param concordance_comprehensive: comprehensive concordance matrix
    :param non_discordance: non-discordance matrix
    :param counter_veto_occurs: counter veto occurs matrix
    :param number_of_criteria: number of criteria

    :return: Credibility matrix with float values from [0, 1] interval
    """
    return pd.DataFrame(
        [
            [
                credibility_cv_pair(
                    concordance_comprehensive[alt_name_b][alt_name_a],
                    non_discordance[alt_name_b][alt_name_a],
                    counter_veto_occurs[alt_name_b][alt_name_a],
                    number_of_criteria,
                )
                for alt_name_b in concordance_comprehensive.columns.values
            ]
            for alt_name_a in concordance_comprehensive.index.values
        ],
        index=concordance_comprehensive.index,
        columns=concordance_comprehensive.columns,
    )


def get_criteria_counts_marginal(
    a_values: pd.Series,
    b_values: pd.Series,
    scales: pd.Series,
    indifference_thresholds: pd.Series,
    preference_thresholds: pd.Series,
) -> pd.Series:
    """Calculates criteria counts for a pair.

    :param a_values: criteria values of first alternative/profile
    :param b_values: criteria values of second alternative/profile
    :param scales: criteria scales with specified preference direction
    :param indifference_thresholds: criteria indifference thresholds
    :param preference_thresholds: criteria performance thresholds

    :return: List of criteria counts for a pair
    """
    np = nq = ni = no = 0

    for i in range(len(a_values)):
        difference = get_criterion_difference(a_values[i], b_values[i], scales[i])

        if difference > 0.0:
            if difference >= preference_thresholds[i](a_values[i]):
                np += 1
            elif difference > indifference_thresholds[i](a_values[i]):
                nq += 1
            else:
                ni += 1
        elif math.isclose(difference, 0.0):
            no += 1

    return pd.Series([np, nq, ni, no])


def get_criteria_counts(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    indifference_thresholds: pd.Series,
    preference_thresholds: pd.Series,
    profiles_perform: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Calculates criteria counts for each pair.

    :param performance_table: performance table (alternatives/profiles as rows, criteria as columns)
    :param scales: criteria scales with specified preference direction
    :param indifference_thresholds: criteria indifference thresholds
    :param preference_thresholds: criteria performance thresholds
    :param profiles_perform: performance table (alternatives/profiles as rows, criteria as columns),
    defaults to None

    :return: Matrix of criteria counts for each pair
    """
    columns_content = (
        profiles_perform if profiles_perform is not None else performance_table
    )

    return pd.DataFrame(
        [
            [
                get_criteria_counts_marginal(
                    performance_table.loc[alt_name_a],
                    columns_content.loc[alt_name_b],
                    scales,
                    indifference_thresholds,
                    preference_thresholds,
                )
                for alt_name_b in columns_content.index.values
            ]
            for alt_name_a in performance_table.index.values
        ],
        index=performance_table.index,
        columns=columns_content.index,
    )


class RelationType(float, Enum):
    SQ = 1.0
    SC = 0.8
    SP = 0.6
    SS = 0.4
    SV = 0.2
    SNone = 0.0


def _calculate_credibility_values(
    performance_table: pd.DataFrame,
    criteria_counts: pd.DataFrame,
    scales: pd.Series,
    veto_thresholds: pd.Series,
    profiles_perform: Optional[pd.DataFrame] = None,
    profiles_criteria_counts: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Determines value and type of relation depending on criteria counts.

    :param performance_table: performance table (alternatives as rows, criteria as columns)
    :param criteria_counts: criteria counts for each pair
    :param scales: criteria scales with specified preference direction
    :param veto_thresholds: criteria veto thresholds
    :param profiles_perform: performance table (alternatives/profiles as rows, criteria as columns),
    defaults to None
    :param profiles_criteria_counts: criteria counts for each pair, defaults to None

    :return: Credibility matrix
    """
    if profiles_perform is None or profiles_criteria_counts is None:
        profiles_perform, profiles_criteria_counts = performance_table, criteria_counts

    credibility_matrix = pd.DataFrame(
        np.zeros(shape=(len(performance_table.index), len(profiles_perform.index))),
        index=performance_table.index,
        columns=profiles_perform.index,
    )

    for alt_name_a in performance_table.index.values:
        for alt_name_b in profiles_perform.index.values:
            if alt_name_a == alt_name_b:
                credibility_matrix[alt_name_b][alt_name_a] = RelationType.SQ
            else:
                np_ab, nq_ab, ni_ab = criteria_counts[alt_name_b][alt_name_a][:3]
                np_ba, nq_ba, ni_ba = profiles_criteria_counts.loc[
                    alt_name_b, alt_name_a
                ][:3]

                if np_ba + nq_ba == 0 and ni_ba < np_ab + nq_ab + ni_ab:
                    credibility_matrix[alt_name_b][alt_name_a] = RelationType.SQ

                elif np_ba == 0:
                    if nq_ba <= np_ab and nq_ba + ni_ba < np_ab + nq_ab + ni_ab:
                        credibility_matrix[alt_name_b][alt_name_a] = RelationType.SC

                    elif nq_ba <= np_ab + nq_ab:
                        credibility_matrix[alt_name_b][alt_name_a] = RelationType.SP

                    else:
                        credibility_matrix[alt_name_b][alt_name_a] = RelationType.SS
                elif (
                    np_ba <= 1
                    and np_ab >= len(scales) // 2
                    and is_veto(
                        profiles_perform.loc[alt_name_b],
                        performance_table.loc[alt_name_a],
                        scales,
                        veto_thresholds,
                    )
                    is False
                ):
                    credibility_matrix[alt_name_b][alt_name_a] = RelationType.SV

                else:
                    credibility_matrix[alt_name_b][alt_name_a] = RelationType.SNone

    return credibility_matrix


def get_credibility_values(
    performance_table: pd.DataFrame,
    criteria_counts: pd.DataFrame,
    scales: pd.Series,
    veto_thresholds: pd.Series,
    profiles_perform: Optional[pd.DataFrame] = None,
    profiles_criteria_counts: Optional[pd.DataFrame] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Determines value and type of relation depending on criteria counts.

    :param performance_table: performance table (alternatives as rows, criteria as columns)
    :param criteria_counts: criteria counts for each pair
    :param scales: criteria scales with specified preference direction
    :param veto_thresholds: criteria veto thresholds
    :param profiles_perform: performance table (alternatives/profiles as rows, criteria as columns),
    defaults to None
    :param profiles_criteria_counts: criteria counts for each pair, defaults to None

    :return: Credibility matrix (or 2 for profiles)
    """
    if profiles_perform is not None and profiles_criteria_counts is not None:
        return _calculate_credibility_values(
            performance_table,
            criteria_counts,
            scales,
            veto_thresholds,
            profiles_perform,
            profiles_criteria_counts,
        ), _calculate_credibility_values(
            profiles_perform,
            profiles_criteria_counts,
            scales,
            veto_thresholds,
            performance_table,
            criteria_counts,
        )

    return _calculate_credibility_values(
        performance_table, criteria_counts, scales, veto_thresholds
    )


def credibility_electre_iv(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    indifference_thresholds: pd.Series,
    preference_thresholds: pd.Series,
    veto_thresholds: pd.Series,
    profiles_perform: Optional[pd.DataFrame] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Computes the credibility of an outranking relation S(a, b) for Electre IV,
    based on performance table (alternatives as rows, criteria as columns),
    scales and thresholds (indifference, preference, veto - Optional),
    also profiles performance table (profiles as rows, criteria as columns) optionally.

    :param performance_table: performance table (alternatives as rows, criteria as columns)
    :param scales: criteria scales with specified preference direction
    :param indifference_thresholds: criteria indifference thresholds
    :param preference_thresholds: criteria performance thresholds
    :param veto_thresholds: criteria veto thresholds
    :param profiles_perform: performance table (profiles as rows, criteria as columns),
    defaults to None

    :return: Credibility matrix
    """
    return get_credibility_values(
        performance_table,
        get_criteria_counts(
            performance_table,
            scales,
            indifference_thresholds,
            preference_thresholds,
            profiles_perform,
        ),
        scales,
        veto_thresholds,
        profiles_perform,
        get_criteria_counts(
            profiles_perform,
            scales,
            indifference_thresholds,
            preference_thresholds,
            performance_table,
        )
        if profiles_perform is not None
        else None,
    )
