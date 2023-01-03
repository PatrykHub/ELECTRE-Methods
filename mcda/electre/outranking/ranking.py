"""This module implements modules to explore outranking with ranking methods."""
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
    """This function computes net flow scores for all
    alternatives.
    :param crisp_outranking_table: table with crisp relations
    between alternatives
    :return: net flow scores for all alternatives
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


def _get_maximal_credibility_index(credibility_matrix: pd.DataFrame) -> NumericValue:
    """Selects the maximal credibility index, based on credibility matrix S(a, b)
    with zeroed diagonal.

    :param credibility_matrix: matrix with credibility values for each alternatives' pair

    :return: Maximal credibility index, value from the [0, 1] interval
    """
    return max(credibility_matrix.max())


def _get_minimal_credibility_index(
    credibility_matrix: pd.DataFrame,
    maximal_credibility_index: NumericValue,
    alpha: NumericValue = -0.15,
    beta: NumericValue = 0.30,
) -> NumericValue:
    """Selects the minimal credibility index, based on credibility matrix S(a, b)
    with zeroed diagonal, maximal_credibility_index and given linear function coefficients.

    :param credibility_matrix: matrix with credibility values for each alternatives' pair
    :param maximal_credibility_index: maximal credibility index, value from the [0, 1] interval
    :param alpha: coefficient of the independent variable, defaults to -0.15
    :param beta: y-intercept, defaults to 0.30

    :return: Minimal credibility index, value from the [0, 1] interval
    """
    threshold_value = maximal_credibility_index - linear_function(
        maximal_credibility_index, alpha, beta
    )
    if threshold_value > maximal_credibility_index:
        raise exceptions.ValueOutsideScaleError(
            "Provided alpha and beta values make it impossible to "
            "calculate a positive function value s = alpha * credibility + beta"
        )

    for alt_name_a in credibility_matrix.index.values:
        for alt_name_b in credibility_matrix.index.values:
            if threshold_value <= credibility_matrix.loc[alt_name_a][alt_name_b]:
                credibility_matrix.loc[alt_name_a][alt_name_b] = 0.0

    return max(credibility_matrix.max())


def crisp_outranking_relation_distillation(
    credibility_pair_value_ab: NumericValue,
    credibility_pair_value_ba: NumericValue,
    minimal_credibility_index: NumericValue,
    alpha: NumericValue = -0.15,
    beta: NumericValue = 0.30,
) -> int:
    """Constructs crisp outranking relations for distillation,
    based on credibility values of pairs (a, b) and (b, a) alternatives,
    also minimal_credibility_index.

    :param credibility_pair_value_ab: credibility value of (a, b) alternatives
    :param credibility_pair_value_ba:  credibility value of (b, a) alternatives
    :param minimal_credibility_index: minimal credibility index, value from the [0, 1] interval
    :param alpha: coefficient of the independent variable, defaults to -0.15
    :param beta: y-intercept, defaults to 0.30

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
    credibility_matrix: pd.DataFrame,
    alpha: NumericValue = -0.15,
    beta: NumericValue = 0.30,
    maximal_credibility_index: Optional[NumericValue] = None,
) -> Tuple[pd.Series, NumericValue]:
    """Computes strength and weakness of each alternative a as the numbers of alternatives
    which are, respectively, outranked by a or outrank a.

    :param credibility_matrix: matrix with credibility values for each alternatives' pair
    :param alpha: coefficient of the independent variable, defaults to -0.15
    :param beta: y-intercept, defaults to 0.30
    :param maximal_credibility_index: optional minimal credibility index from outer distillation,
    defaults to None

    :return: Quality of a computed as the difference of its strength and weakness,
    also maximal credibility index for possible inner distillation
    """
    _consistent_df_indexing(credibility_matrix=credibility_matrix)
    if set(credibility_matrix.columns) != set(credibility_matrix.index):
        raise exceptions.InconsistentDataFrameIndexingError(
            "Quality calculation is possible only for alternatives, but "
            "for the provided credibility table, the set of values "
            "in rows is different than in the columns."
        )

    if maximal_credibility_index is None:
        maximal_credibility_index = _get_maximal_credibility_index(credibility_matrix)
    minimal_credibility_index = _get_minimal_credibility_index(
        credibility_matrix.copy(), maximal_credibility_index, alpha, beta
    )

    alternatives_strength = pd.Series(
        {
            alt_name_a: sum(
                (
                    crisp_outranking_relation_distillation(
                        credibility_matrix.loc[alt_name_a][alt_name_b],
                        credibility_matrix.loc[alt_name_b][alt_name_a],
                        minimal_credibility_index,
                        alpha,
                        beta,
                    )
                )
                for alt_name_b in credibility_matrix.index.values
            )
            for alt_name_a in credibility_matrix.index.values
        }
    )

    alternatives_weakness = pd.Series(
        {
            alt_name_b: sum(
                (
                    crisp_outranking_relation_distillation(
                        credibility_matrix.loc[alt_name_a][alt_name_b],
                        credibility_matrix.loc[alt_name_b][alt_name_a],
                        minimal_credibility_index,
                        alpha,
                        beta,
                    )
                )
                for alt_name_a in credibility_matrix.index.values
            )
            for alt_name_b in credibility_matrix.index.values
        }
    )

    return alternatives_strength - alternatives_weakness, minimal_credibility_index


def _distillation_process(
    credibility_matrix: pd.DataFrame,
    remaining_alt_indices: pd.Series,
    preference_operator: Callable,
    alpha: NumericValue = -0.15,
    beta: NumericValue = 0.30,
    maximal_credibility_index: Optional[NumericValue] = None,
) -> Tuple[pd.Series, NumericValue]:
    """Conducts main distillation process.

    :param credibility_matrix: matrix with credibility values for each alternatives' pair
    :param remaining_alt_indices: remaining alternatives' indices to conduct distillation
    :param preference_operator: represents distillation order (max - downward / min - upward)
    :param alpha: coefficient of the independent variable, defaults to -0.15
    :param beta: y-intercept, defaults to 0.30
    :param maximal_credibility_index: optional minimal credibility index from outer distillation,
    defaults to None

    :return: Set of alternatives with the greatest quality,
    also maximal credibility index for possible inner distillation
    """
    updated_credibility_matrix = credibility_matrix.loc[remaining_alt_indices][
        remaining_alt_indices
    ]

    qualities, minimal_credibility_index = alternative_qualities(
        updated_credibility_matrix, alpha, beta, maximal_credibility_index
    )

    return (
        qualities[qualities == preference_operator(qualities)],
        minimal_credibility_index,
    )


def distillation(
    credibility_matrix: pd.DataFrame,
    upward_order: bool = False,
    alpha: NumericValue = -0.15,
    beta: NumericValue = 0.30,
) -> pd.Series:
    """Conducts either descending or ascending distillation in the set of alternatives
    on the basis of credibility matrix. Depending on the boolean variable upward order,
    it provides either upward or downward order. Output can be parametrized
    with linear function coefficients.

    :param credibility_matrix: matrix with credibility values for each alternatives' pair
    :param upward_order: descending order if False, otherwise ascending, defaults to False
    :param alpha: coefficient of the independent variable, defaults to -0.15
    :param beta: y-intercept, defaults to 0.30

    :return: Nested list of complete upward or downward order
    """
    try:
        np.fill_diagonal(credibility_matrix.values, 0)
        preference_operator = min if upward_order else max
        remaining_alt_indices = credibility_matrix.index.to_series()
    except (TypeError, AttributeError) as exc:
        exc.args = (
            f"Wrong credibility matrix type. Expected {pd.DataFrame.__name__}, "
            f"but got {type(credibility_matrix).__name__} instead.",
        )
        raise
    order = pd.Series([], dtype="float64")
    level: int = 1

    while not remaining_alt_indices.empty:
        preferred_alternatives, minimal_credibility_index = _distillation_process(
            credibility_matrix, remaining_alt_indices, preference_operator, alpha, beta
        )

        # inner distillation procedure
        while len(preferred_alternatives) > 1 and minimal_credibility_index > 0:
            preferred_alternatives, minimal_credibility_index = _distillation_process(
                credibility_matrix,
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
    descending_order: pd.Series, ascending_order: pd.Series
) -> pd.DataFrame:
    """Constructs final partial preorder intersection from downward and upward orders of
    alternatives derived from the descending and ascending distillation procedures, respectively.

    :param descending_order_matrix: outranking matrix from downward order
    :param ascending_order_matrix: outranking matrix from upward order

    :return: Final outranking matrix
    """
    descending_order_matrix = order_to_outranking_matrix(descending_order)
    ascending_order_matrix = order_to_outranking_matrix(ascending_order)
    _consistent_df_indexing(
        descending_order_matrix=descending_order_matrix,
        ascending_order_matrix=ascending_order_matrix,
    )
    return descending_order_matrix * ascending_order_matrix


def ranks(final_ranking_matrix: pd.DataFrame) -> pd.Series:
    """Constructs ranks of the alternatives in the final preorder.

    :param final_ranking_matrix: outranking matrix from final ranking

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
    """Constructs median preorder.

    :param ranks: nested list of ranks of the alternatives
    :param downward_order: nested list of downward order
    :param upward_order: nested list of upward order

    :return: Nested list of median preorder
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
