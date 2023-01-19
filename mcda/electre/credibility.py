"""This module implements methods to compute a valued outranking relation
which can be interpreted as a degree of credibility of an outranking relation.

Implementation based on:
    * credibility_comprehensive: :cite:p:`DelVastoTerrientes15`,
    * credibility_cv: :cite:p:`Roy08`,
    * credibility_electre_iv: :cite:p:`Wolny15`.
"""
import math
from enum import Enum
from typing import Optional, Sized, Tuple, Union

import numpy as np
import pandas as pd

from ..core.aliases import NumericValue
from . import exceptions
from ._validation import (
    _check_df_index,
    _check_index_value_interval,
    _consistent_criteria_names,
    _consistent_df_indexing,
    _get_threshold_values,
)
from .utils import get_criterion_difference, is_veto_exceeded


def credibility_pair(
    concordance_comprehensive: NumericValue,
    non_discordance: NumericValue,
) -> NumericValue:
    """Computes the credibility value math:`S(a, b)` of an outranking relation, based on
    comprehensive concordance :math:`C(a, b)` index multiplied by non-discordance
    :math:`{\\Delta}(a, b)`.

    :param concordance_comprehensive: comprehensive concordance value C(a, b)
    :param non_discordance: non-discordance value :math:`{\\Delta}(a,b)`

    :return: credibility value, from [0, 1] interval
    """
    _check_index_value_interval(concordance_comprehensive, name="comprehensive concordance")
    _check_index_value_interval(non_discordance, name="discordance index")
    return concordance_comprehensive * non_discordance


def credibility_comprehensive(
    concordance_comprehensive: pd.DataFrame,
    discordance: pd.DataFrame,
    is_non_discordance: bool = True,
) -> pd.DataFrame:
    """Computes the credibility :math:`S` of an outranking relation, based on
    comprehensive concordance :math:`C` multiplied by non-discordance :math:`{\\Delta}`
    or comprehensive discordance :math:`D`.

    :param concordance_comprehensive: comprehensive concordance :math:`C`
    :param discordance: discordance :math:`D` or non-discordance :math:`{\\Delta}`

    :param is_non_discordance: ``True`` if non-discordance is provided, otherwise ``False``,
        defaults to ``True``

    :return: credibility table :math:`S` with values from [0, 1] interval
    """
    _consistent_df_indexing(
        concordance_comprehensive=concordance_comprehensive,
        discordance=discordance,
    )
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
    """Computes the single credibility value :math:`S^{CV}(a, b)` of an outranking relation
    with counter-veto effect, based on comprehensive concordance :math:`C(a, b)`
    and non-discordance :math:`{\\Delta}(a, b)` indices, also counter-veto occurs :math:`cv(a, b)`
    with the number of criteria.

    :param concordance_comprehensive: comprehensive concordance index :math:`C(a, b)`
    :param non_discordance: non-discordance index :math:`{\\Delta} (a, b)`
    :param counter_veto_occurs: counter-veto occurs list or value :math:`cv(a, b)`
    :param number_of_criteria: number of criteria

    :raises TypeError: _description_
    :raises ValueError: _description_
    :raises TypeError: _description_
    :raises ValueError: _description_
    .. todo::
        describe errors

    :return: credibility value :math:`S^{CV}(a, b)` with counter-veto effect,
        value from [0, 1] interval
    """
    _check_index_value_interval(concordance_comprehensive, name="comprehensive concordance")
    _check_index_value_interval(non_discordance, name="non-discordance index")

    try:
        if int(number_of_criteria) != number_of_criteria:
            raise TypeError

        if number_of_criteria <= 0:
            raise ValueError(
                "Number of criteria must be a positive number, but "
                f"got {number_of_criteria} instead."
            )
    except TypeError as exc:
        exc.args = (
            "Number of criteria should be an integer, but got "
            f"{type(number_of_criteria).__name__} instead.",
        )
        raise

    try:
        if not isinstance(counter_veto_occurs, Sized):
            if int(counter_veto_occurs) != counter_veto_occurs:
                raise TypeError

            if counter_veto_occurs < 0:
                raise ValueError(
                    "Counter veto occurrence count cannot be less than 0, "
                    f"but got {counter_veto_occurs} instead."
                )
        return concordance_comprehensive * non_discordance ** (
            1
            - (
                len(counter_veto_occurs)
                if isinstance(counter_veto_occurs, Sized)
                else counter_veto_occurs
            )
            / number_of_criteria
        )
    except TypeError as exc:
        exc.args = (
            "Wrong argument type with cv occurrence information. "
            f"Expected {credibility_cv_pair.__annotations__['counter_veto_occurs']}, "
            f"but got {type(counter_veto_occurs).__name__} instead.",
        )
        raise


def credibility_cv(
    concordance_comprehensive: pd.DataFrame,
    non_discordance: pd.DataFrame,
    counter_veto_occurs: pd.DataFrame,
    number_of_criteria: int,
) -> pd.DataFrame:
    """Computes the credibility :math:`S^{CV}` of an outranking relation with counter-veto effect,
    based on comprehensive concordance :math:`C` and non-discordance :math:`{\\Delta}`,
    also counter-veto occurs :math:`cv` with the number of criteria.

    :param concordance_comprehensive: comprehensive concordance :math:`C`
    :param non_discordance: non-discordance value :math:`{\\Delta}`
    :param counter_veto_occurs: counter-veto occurs with lists or values :math:`cv(a, b)`
    :param number_of_criteria: number of criteria

    :return: credibility table :math:`S^{CV}` with values from [0, 1] interval
    """
    _consistent_df_indexing(
        concordance_comprehensive=concordance_comprehensive,
        non_discordance=non_discordance,
        counter_veto_occurs=counter_veto_occurs,
    )
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
    **kwargs,
) -> pd.Series:
    """Computes criteria parameters for a pair between alternatives or alternatives-profiles.

    :param a_values: alternative's performance on all its criteria
    :param b_values: alternative's performance on all its criteria
    :param scales: all criteria's scales with specified preference direction
    :param indifference_thresholds: all criteria's indifference thresholds
    :param preference_thresholds: all criteria's preference thresholds

    :raises exceptions.WrongThresholdValueError: _description_
    .. todo::
        describe raised exception

    :return: criteria counts for a pair
    """
    np = nq = ni = no = 0
    if "validated" not in kwargs:
        _consistent_criteria_names(
            a_values=a_values,
            b_values=b_values,
            scales=scales,
            indifference_thresholds=indifference_thresholds,
            preference_thresholds=preference_thresholds,
        )

    for i in range(len(a_values)):
        difference = get_criterion_difference(a_values[i], b_values[i], scales[i])
        indifference, preference = _get_threshold_values(
            a_values[i],
            indifference_threshold=indifference_thresholds[i],
            preference_threshold=preference_thresholds[i],
        )
        if not 0 <= indifference <= preference:
            raise exceptions.WrongThresholdValueError(
                "Threshold values must meet a condition: "
                "0 <= indifference_threshold <= preference_threshold, but got "
                f"0 <= {indifference} <= {preference}"
            )

        if difference > 0.0:
            if difference >= preference:
                np += 1
            elif difference > indifference:
                nq += 1
            else:
                ni += 1
        elif math.isclose(difference, 0.0):
            no += 1

    return pd.Series([np, nq, ni, no], index=["np", "nq", "ni", "no"])


def get_criteria_counts(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    indifference_thresholds: pd.Series,
    preference_thresholds: pd.Series,
    profiles_perform: Optional[pd.DataFrame] = None,
    **kwargs,
) -> pd.DataFrame:
    """Computes criteria parameters for each pair between alternatives or alternatives-profiles.

    :param performance_table: performance table (alternatives/profiles as rows, criteria as columns)
    :param scales: all criteria's scales with specified preference direction
    :param indifference_thresholds: all criteria's indifference thresholds
    :param preference_thresholds: all criteria's preference thresholds
    :param profiles_perform: optional profiles performance table (alternatives/profiles as rows,
        criteria as columns), defaults to ``None``

    :return: criteria counts for each pair
    """
    if "validated" not in kwargs:
        _consistent_criteria_names(
            performance_table=performance_table,
            scales=scales,
            indifference_thresholds=indifference_thresholds,
            preference_thresholds=preference_thresholds,
            profiles_perform=profiles_perform,
        )
        _check_df_index(performance_table, index_type="alternatives")
        _check_df_index(profiles_perform, index_type="criteria")

    columns_content = profiles_perform if profiles_perform is not None else performance_table
    return pd.DataFrame(
        [
            [
                get_criteria_counts_marginal(
                    performance_table.loc[alt_name_a],
                    columns_content.loc[alt_name_b],
                    scales,
                    indifference_thresholds,
                    preference_thresholds,
                    validated=True,
                )
                for alt_name_b in columns_content.index.values
            ]
            for alt_name_a in performance_table.index.values
        ],
        index=performance_table.index,
        columns=columns_content.index,
    )


class RelationType(float, Enum):
    """Definition of five predefined outranking relations in ELECTRE IV credibility.

    :param float: credibility value of predefined relation
    :param Enum: shortcut of predefined relation name

    .. note::
        Relations from the most to the least credible:
        :math:`SQ` - quasi-dominance,
        :math:`SC` - canonic-dominance,
        :math:`SP` - pseudo-dominance,
        :math:`SS` - sub-dominance,
        :math:`SV` - veto-dominance.
    """

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

    :param performance_table: performance table (alternatives/profiles as rows, criteria as columns)
    :param criteria_counts: criteria counts for each pair
    :param scales: all criteria's scales with specified preference direction
    :param veto_thresholds: all criteria's veto thresholds
    :param profiles_perform: optional profiles performance table (alternatives/profiles as rows,
        criteria as columns), defaults to ``None``
    :param profiles_criteria_counts: optional criteria counts for each pair, defaults to ``None``

    :return: credibility :math:`S^{IV}` with values from [0, 1] interval
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
                np_ba, nq_ba, ni_ba = profiles_criteria_counts.loc[alt_name_b, alt_name_a][:3]

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
                    and is_veto_exceeded(
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


def _get_credibility_values(
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
    :param scales: all criteria's scales with specified preference direction
    :param veto_thresholds: all criteria's veto thresholds
    :param profiles_perform: optional profiles performance table (alternatives/profiles as rows,
        criteria as columns), defaults to ``None``
    :param profiles_criteria_counts: optional criteria counts for each pair, defaults to ``None``

    :return: credibility :math:`S^{IV}` (or 2 for profiles) with values from [0, 1] interval
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
    """Computes the credibility of an outranking relation :math:`S^{CV}` for alternatives
    or alternatives-profiles, based on performance table, scales, the indifference, preference,
    veto thresholds, and optionally profiles performance table.

    :param performance_table: performance table (alternatives as rows, criteria as columns)
    :param scales: all criteria's scales with specified preference direction
    :param indifference_thresholds: all criteria's indifference thresholds
    :param preference_thresholds: all criteria's preference thresholds
    :param veto_thresholds: all criteria's veto thresholds
    :param profiles_perform: optional profiles performance table (alternatives/profiles as rows,
        criteria as columns), defaults to ``None``

    :return:
        * if `profiles_perform` argument is set to ``None``, the function will
          return a single `pandas.DataFrame` object with :math:`S^{IV}(a, b)` values
          for all alternatives pairs
        * otherwise, the function will return a ``tuple`` object with two
          `pandas.DataFrame` objects inside, where the first one contains
          :math:`S^{IV}(alternative_i, profile_j)` indices, and the second one
          :math:`S^{IV}(profile_j, alternative_i)`, respectively
    """
    _consistent_criteria_names(
        performance_table=performance_table,
        scales=scales,
        indifference_thresholds=indifference_thresholds,
        preference_thresholds=preference_thresholds,
        veto_thresholds=veto_thresholds,
        profiles_perform=profiles_perform,
    )
    _check_df_index(performance_table, index_type="alternatives")
    _check_df_index(profiles_perform, index_type="criteria")

    return _get_credibility_values(
        performance_table,
        get_criteria_counts(
            performance_table,
            scales,
            indifference_thresholds,
            preference_thresholds,
            profiles_perform,
            validated=True,
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
            validated=True,
        )
        if profiles_perform is not None
        else None,
    )
