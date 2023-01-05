"""This module implements methods to explore outranking relations with ranking problems."""
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from mcda.core.aliases import NumericValue

from .. import exceptions
from .._validation import (
    _check_index_value_binary,
    _check_index_value_interval,
    _consistent_df_indexing,
    _unique_names,
)
from ..utils import (
    linear_function,
    order_to_outranking_matrix,
    reverse_transform_series,
    transform_series,
)


def net_flow_score(outranking_table: pd.DataFrame) -> pd.Series:
    """Constructs descending list of alternatives ordered by their Net Flow Score,
    based on an outranking table.

    :param outranking_table: crisp or valued outranking table DataFrame

    :raises exceptions.InconsistentDataFrameIndexingError: _description_
    .. todo::
        describe exception

    :return: Ordered Series with Net Flow Scores for all alternatives
    """
    _consistent_df_indexing(outranking_table=outranking_table)
    if set(outranking_table.columns) != set(outranking_table.index):
        raise exceptions.InconsistentDataFrameIndexingError(
            "NFS calculation is possible only for alternatives, but "
            "for the provided outranking table, the set of values "
            "in rows is different than in the columns."
        )
    for column_name in outranking_table.columns.values:
        for row_name in outranking_table.index.values:
            _check_index_value_interval(
                outranking_table[column_name][row_name], name="outranking relation"
            )
    return pd.Series(
        [
            outranking_table.loc[alt_name].sum() - outranking_table[alt_name].sum()
            for alt_name in outranking_table.index.values
        ],
        index=outranking_table.index,
    ).sort_values(ascending=False)


def _get_maximal_credibility_index(credibility_table: pd.DataFrame) -> NumericValue:
    """Selects the maximal credibility index, based on credibility table :math:`S` of an outranking
    relation with zeroed diagonal.

    :param credibility_table: credibility :math:`S` DataFrame for each alternatives' pair

    :return:  Maximal credibility index, value from the :math:`[0, 1]` interval
    """
    return max(credibility_table.max())


def _get_minimal_credibility_index(
    credibility_table: pd.DataFrame,
    maximal_credibility_index: NumericValue,
    alpha: NumericValue = -0.15,
    beta: NumericValue = 0.30,
) -> NumericValue:
    """Selects the minimal credibility index, based on credibility table :math:`S` of an outranking
    relation with zeroed diagonal, maximal credibility index and given linear function coefficients.

    :param credibility_table: credibility :math:`S` DataFrame for each alternatives' pair
    :param maximal_credibility_index: maximal credibility index,
        value from the :math:`[0, 1]` interval
    :param alpha: coefficient of the independent variable, defaults to -0.15
    :param beta: y-intercept, defaults to 0.30

    :raises exceptions.ValueOutsideScaleError: _description_
    .. todo::
        describe exception

    :return: Minimal credibility index, value from the :math:`[0, 1]` interval
    """
    threshold_value = maximal_credibility_index - linear_function(
        maximal_credibility_index, alpha, beta
    )
    if threshold_value > maximal_credibility_index:
        raise exceptions.ValueOutsideScaleError(
            "Provided alpha and beta values make it impossible to "
            "calculate a positive function value s = alpha * credibility + beta"
        )

    for alt_name_a in credibility_table.index.values:
        for alt_name_b in credibility_table.index.values:
            if threshold_value <= credibility_table.loc[alt_name_a][alt_name_b]:
                credibility_table.loc[alt_name_a][alt_name_b] = 0.0

    return max(credibility_table.max())


def crisp_outranking_relation_distillation(
    credibility_pair_value_ab: NumericValue,
    credibility_pair_value_ba: NumericValue,
    minimal_credibility_index: NumericValue,
    alpha: NumericValue = -0.15,
    beta: NumericValue = 0.30,
) -> int:
    """Constructs crisp outranking relations for distillation,
    based on credibility values of pairs (a, b) and (b, a) alternatives,
    also minimal credibility index.

    :param credibility_pair_value_ab: credibility value of (a, b) alternatives
    :param credibility_pair_value_ba: credibility value of (b, a) alternatives
    :param minimal_credibility_index: minimal credibility index, value from the [0, 1] interval
    :param alpha: coefficient of the independent variable, defaults to -0.15
    :param beta: y-intercept, defaults to 0.30

    :raises exceptions.ValueOutsideScaleError: _description_
    .. todo::
        describe exception

    :return: 1 if undermentioned inequality is true, 0 otherwise
    """
    _check_index_value_interval(credibility_pair_value_ab, name="credibility")
    _check_index_value_interval(credibility_pair_value_ba, name="credibility")
    _check_index_value_interval(
        minimal_credibility_index, name="minimal credibility index"
    )

    difference_threshold = linear_function(alpha, credibility_pair_value_ab, beta)
    if difference_threshold < 0:
        raise exceptions.ValueOutsideScaleError(
            "Provided alpha and beta values make it impossible to "
            "calculate a positive function value s = alpha * credibility + beta"
        )
    return (
        1
        if credibility_pair_value_ab > minimal_credibility_index
        and credibility_pair_value_ab > credibility_pair_value_ba + difference_threshold
        else 0
    )


def alternative_qualities(
    credibility_table: pd.DataFrame,
    alpha: NumericValue = -0.15,
    beta: NumericValue = 0.30,
    maximal_credibility_index: Optional[NumericValue] = None,
) -> Tuple[pd.Series, NumericValue]:
    """Computes strength and weakness of each alternative a as the numbers of alternatives
    which are, respectively, outranked by a or outrank a.

    :param credibility_table: credibility :math:`S` DataFrame for each alternatives' pair
    :param alpha: coefficient of the independent variable, defaults to -0.15
    :param beta: y-intercept, defaults to 0.30
    :param maximal_credibility_index: optional minimal credibility index from outer distillation,
        defaults to None

    :raises exceptions.InconsistentDataFrameIndexingError: _description_
    .. todo::
        describe exception

    :return: Quality of a computed as the difference of its strength and weakness,
        also maximal credibility index for possible inner distillation
    """
    _consistent_df_indexing(credibility_matrix=credibility_table)
    if set(credibility_table.columns) != set(credibility_table.index):
        raise exceptions.InconsistentDataFrameIndexingError(
            "Quality calculation is possible only for alternatives, but "
            "for the provided credibility table, the set of values "
            "in rows is different than in the columns."
        )

    if maximal_credibility_index is None:
        maximal_credibility_index = _get_maximal_credibility_index(credibility_table)
    minimal_credibility_index = _get_minimal_credibility_index(
        credibility_table.copy(), maximal_credibility_index, alpha, beta
    )

    alternatives_strength = pd.Series(
        {
            alt_name_a: sum(
                (
                    crisp_outranking_relation_distillation(
                        credibility_table.loc[alt_name_a][alt_name_b],
                        credibility_table.loc[alt_name_b][alt_name_a],
                        minimal_credibility_index,
                        alpha,
                        beta,
                    )
                )
                for alt_name_b in credibility_table.index.values
            )
            for alt_name_a in credibility_table.index.values
        }
    )

    alternatives_weakness = pd.Series(
        {
            alt_name_b: sum(
                (
                    crisp_outranking_relation_distillation(
                        credibility_table.loc[alt_name_a][alt_name_b],
                        credibility_table.loc[alt_name_b][alt_name_a],
                        minimal_credibility_index,
                        alpha,
                        beta,
                    )
                )
                for alt_name_a in credibility_table.index.values
            )
            for alt_name_b in credibility_table.index.values
        }
    )

    return alternatives_strength - alternatives_weakness, minimal_credibility_index


def _distillation_process(
    credibility_table: pd.DataFrame,
    remaining_alt_indices: pd.Series,
    preference_operator: Callable,
    alpha: NumericValue = -0.15,
    beta: NumericValue = 0.30,
    maximal_credibility_index: Optional[NumericValue] = None,
) -> Tuple[pd.Series, NumericValue]:
    """Conducts main distillation process.

    :param credibility_table: credibility :math:`S` DataFrame for each alternatives' pair
    :param remaining_alt_indices: remaining alternatives' indices to conduct distillation
    :param preference_operator: represents distillation order (max - downward / min - upward)
    :param alpha: coefficient of the independent variable, defaults to -0.15
    :param beta: y-intercept, defaults to 0.30
    :param maximal_credibility_index: optional minimal credibility index from outer distillation,
        defaults to None

    :return: Set of alternatives with the greatest quality,
        also maximal credibility index for possible inner distillation
    """
    updated_credibility_table = credibility_table.loc[remaining_alt_indices][
        remaining_alt_indices
    ]

    qualities, minimal_credibility_index = alternative_qualities(
        updated_credibility_table, alpha, beta, maximal_credibility_index
    )

    return (
        qualities[qualities == preference_operator(qualities)],
        minimal_credibility_index,
    )


def distillation(
    credibility_table: pd.DataFrame,
    upward_order: bool = False,
    alpha: NumericValue = -0.15,
    beta: NumericValue = 0.30,
) -> pd.Series:
    """Conducts either descending or ascending distillation from the set of alternatives
    based on the credibility table :math:`S`. Depending on the boolean variable upward order,
    it provides either upward :math:`P^A` or downward :math:`P^D` order. Output can be parametrized
    with linear function coefficients.

    :param credibility_table: credibility :math:`S` DataFrame for each alternatives' pair
    :param upward_order: descending order if False, otherwise ascending, defaults to False
    :param alpha: coefficient of the independent variable, defaults to -0.15
    :param beta: y-intercept, defaults to 0.30

    :return: Nested list of complete upward :math:`P^A` or downward :math:`P^D` order
    """
    try:
        np.fill_diagonal(credibility_table.values, 0)
        preference_operator = min if upward_order else max
        remaining_alt_indices = credibility_table.index.to_series()
    except (TypeError, AttributeError) as exc:
        exc.args = (
            f"Wrong credibility table type. Expected {pd.DataFrame.__name__}, "
            f"but got {type(credibility_table).__name__} instead.",
        )
        raise
    order = pd.Series([], dtype="float64")
    level: int = 1

    while not remaining_alt_indices.empty:
        preferred_alternatives, minimal_credibility_index = _distillation_process(
            credibility_table, remaining_alt_indices, preference_operator, alpha, beta
        )

        # inner distillation procedure
        while len(preferred_alternatives) > 1 and minimal_credibility_index > 0:
            preferred_alternatives, minimal_credibility_index = _distillation_process(
                credibility_table,
                preferred_alternatives.index.to_series(),
                preference_operator,
                alpha,
                beta,
                minimal_credibility_index,
            )

        remaining_alt_indices = remaining_alt_indices.drop(preferred_alternatives.index)
        order[level] = preferred_alternatives.index.to_list()
        level += 1

    return order[::-1] if upward_order else order


def final_ranking_matrix(
    downward_order: pd.Series, upward_order: pd.Series
) -> pd.DataFrame:
    """Constructs final partial preorder intersection :math:`P` from downward :math:`P^D`
    and upward :math:`P^A` orders of alternatives derived from the descending
    and ascending distillation procedures, respectively.

    :param downward_order: downward order :math:`P^D` from distillation process
    :param upward_order: upward order :math:`P^A` from distillation process

    :return: Final outranking matrix :math:`P`
    """
    descending_order_matrix = order_to_outranking_matrix(downward_order)
    ascending_order_matrix = order_to_outranking_matrix(upward_order)
    _consistent_df_indexing(
        descending_order_matrix=descending_order_matrix,
        ascending_order_matrix=ascending_order_matrix,
    )
    return descending_order_matrix * ascending_order_matrix


def ranks(final_ranking_matrix: pd.DataFrame) -> pd.Series:
    """Constructs ranks of the alternatives in the final preorder,
    based on the final ranking matrix.

    :param final_ranking_matrix: outranking matrix from final ranking :math:`P`

    :raises exceptions.InconsistentDataFrameIndexingError: _description_
    .. todo::
        describe exception

    :return: Nested list of ranks
    """
    _consistent_df_indexing(final_ranking_matrix=final_ranking_matrix)
    if set(final_ranking_matrix.columns.values) != set(
        final_ranking_matrix.index.values
    ):
        raise exceptions.InconsistentDataFrameIndexingError(
            "Ranks calculation is possible only for alternatives, but "
            "for the provided outranking table, the set of values "
            "in rows is different than in the columns."
        )
    for column_name in final_ranking_matrix.columns.values:
        for row_name in final_ranking_matrix.index.values:
            _check_index_value_binary(
                final_ranking_matrix[column_name][row_name], name="final ranking index"
            )

    ranks_ranking = pd.Series([], dtype="float64")
    remaining_alt_indices = final_ranking_matrix.index.to_series()
    level: int = 1

    while not remaining_alt_indices.empty:
        rank_level = []
        for alt_name_a in remaining_alt_indices.index.values:
            current_rank = True
            for alt_name_b in remaining_alt_indices.index.values:
                if (
                    alt_name_a != alt_name_b
                    and final_ranking_matrix.loc[alt_name_b][alt_name_a] == 1
                    and final_ranking_matrix.loc[alt_name_a][alt_name_b] == 0
                ):
                    current_rank = False
                    break

            if current_rank:
                rank_level.append(alt_name_a)

        remaining_alt_indices = remaining_alt_indices.drop(rank_level)
        ranks_ranking[level] = rank_level
        level += 1

    return ranks_ranking


def median_order(
    ranks: pd.Series, downward_order: pd.Series, upward_order: pd.Series
) -> pd.Series:
    """Constructs median preorder, based on downward and upward orders
    from distillation process.

    :param ranks: nested list of ranks of the alternatives
    :param downward_order: nested list of downward order :math:`P^D`
    :param upward_order: nested list of upward order :math:`P^A`

    :raises TypeError: _description_
    :raises exceptions.InconsistentIndexNamesError: _description_
    :raises exceptions.InconsistentIndexNamesError: _description_
    :raises exceptions.InconsistentIndexNamesError: _description_
    :raises exceptions.InconsistentIndexNamesError: _description_
    .. todo::
        describe exception

    :return: _description_
    """
    try:
        alternatives = ranks.explode().to_list()
    except AttributeError as exc:
        raise TypeError(
            f"Wrong ranks argument type. Expected {pd.Series.__name__}, "
            f"but got {type(ranks).__name__} instead."
        ) from exc

    ranks = transform_series(ranks)
    downward_order = transform_series(downward_order)
    upward_order = transform_series(upward_order)

    _unique_names(ranks.keys(), names_type="alternatives")
    _unique_names(downward_order.keys(), names_type="alternatives")
    _unique_names(upward_order.keys(), names_type="alternatives")

    ranks_set, downwards_set, upward_set = (
        set(ranks),
        set(downward_order),
        set(upward_order),
    )
    if sum(ranks_set) != sum([x for x in range(1, len(ranks_set) + 1)]):
        raise exceptions.InconsistentIndexNamesError(
            "Values in ranks should be a sequential integers, "
            f"starting with 0, but got {ranks_set} instead."
        )
    if sum(downwards_set) != sum([x for x in range(1, len(downwards_set) + 1)]):
        raise exceptions.InconsistentIndexNamesError(
            "Values in the downward order should be "
            "a sequential integers, starting with 0, but got "
            f"{downwards_set} instead."
        )
    if sum(upward_set) != sum([x for x in range(1, len(upward_set) + 1)]):
        raise exceptions.InconsistentIndexNamesError(
            "Values in the upward order should be "
            "a sequential integers, starting with 0, but got "
            f"{upward_set} instead."
        )

    base_keys_set = set(ranks.keys())
    if base_keys_set != set(downward_order.keys()) or base_keys_set != set(
        upward_order.keys()
    ):
        raise exceptions.InconsistentIndexNamesError(
            "Provided ranks and nested lists with order must contain "
            "the same set of alternatives."
        )

    initial_order = []
    for i in range(len(ranks)):
        initial_order.append(i)
        for j in range(i, 0, -1):
            alt_name_a, alt_name_b = (
                alternatives[initial_order[j]],
                alternatives[initial_order[j - 1]],
            )

            if ranks[alt_name_a] < ranks[alt_name_b]:
                initial_order[j], initial_order[j - 1] = (
                    initial_order[j - 1],
                    initial_order[j],
                )

            elif ranks[alt_name_a] == ranks[alt_name_b]:
                downwards_difference = (
                    downward_order[alt_name_a] - downward_order[alt_name_b]
                )
                upwards_difference = upward_order[alt_name_a] - upward_order[alt_name_b]

                if downwards_difference + upwards_difference < 0:
                    initial_order[j], initial_order[j - 1] = (
                        initial_order[j - 1],
                        initial_order[j],
                    )

    final_order = pd.Series([], dtype="float64")
    level: int = 1
    for i in range(len(initial_order)):
        alt_name_a, alt_name_b = (
            alternatives[initial_order[i]],
            alternatives[initial_order[i - 1]],
        )

        if ranks[alt_name_a] > ranks[alt_name_b]:
            level += 1

        elif ranks[alt_name_a] == ranks[alt_name_b]:
            downwards_difference = (
                downward_order[alt_name_a] - downward_order[alt_name_b]
            )
            upwards_difference = upward_order[alt_name_a] - upward_order[alt_name_b]

            if downwards_difference + upwards_difference > 0:
                level += 1

        final_order[alternatives[initial_order[i]]] = level

    return reverse_transform_series(final_order)
