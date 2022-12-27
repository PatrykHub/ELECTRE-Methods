"""This module implements methods to make an outranking."""
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.aliases import NumericValue
from ._validate import _check_index_value_interval
from .utils import linear_function, reverse_transform_series, transform_series


class OutrankingRelation(Enum):
    INDIFF = "INDIFFERENCE"
    PQ = "STRONG OR WEAK PREFERENCE"
    R = "INCOMPARABILITY"


def crisp_outranking_cut_marginal(
    credibility: NumericValue,
    cutting_level: NumericValue,
) -> bool:
    """Constructs a crisp outranking relation,
    based on the credibility value S(a, b).

    :param credibility: credibility of outranking, value from [0, 1] interval
    :param cutting_level: value from [0.5, 1] interval

    :return: ``True`` if a outranks b, ``False`` otherwise
    """
    _check_index_value_interval(credibility, "credibility")
    _check_index_value_interval(cutting_level, "cutting level", minimal_val=0.5)
    return credibility >= cutting_level


def crisp_outranking_cut(
    credibility_table: pd.DataFrame,
    cutting_level: NumericValue,
) -> pd.DataFrame:
    """Constructs crisp outranking relations,
    based on credibility values.

    :param credibility_table: table with credibility values
    :param cutting_level: value from [0.5, 1] interval

    :return: Boolean table the same size as the credibility table
    """
    return pd.DataFrame(
        [
            [
                crisp_outranking_cut_marginal(
                    credibility_table.loc[alt_name_a][alt_name_b], cutting_level
                )
                for alt_name_b in credibility_table.index.values
            ]
            for alt_name_a in credibility_table.index.values
        ],
        index=credibility_table.index,
        columns=credibility_table.index,
    )


def crisp_outranking_Is_marginal(
    concordance_comprehensive: NumericValue,
    discordance_comprehensive_bin: int,
    concordance_cutting_level: NumericValue,
) -> bool:
    """Constructs a crisp outranking relation, based on
    comprehensive concordance C(a, b) and comprehensive
    binary discordance D(a, b) indices.

    :param concordance_comprehensive: comprehensive concordance of two alternatives
    :param discordance_comprehensive_bin: comprehensive binary concordance of two alternatives
    :param concordance_cutting_level: concordance majority threshold (cutting level)

    :return: ``True`` if a outranks b, ``False`` otherwise
    """
    _check_index_value_interval(concordance_comprehensive, "comprehensive concordance")
    _check_index_value_interval(concordance_cutting_level, "cutting level", minimal_val=0.5)

    if discordance_comprehensive_bin not in [0, 1]:
        raise ValueError(
            "Provided comprehensive discordance is not binary. Expected 0 or 1 value, but "
            f"got {discordance_comprehensive_bin} instead."
        )

    return (
        concordance_comprehensive >= concordance_cutting_level
        and discordance_comprehensive_bin == 0
    )


def crisp_outranking_Is(
    concordance_comprehensive_table: pd.DataFrame,
    discordance_comprehensive_bin_table: pd.DataFrame,
    concordance_cutting_level: NumericValue,
) -> pd.DataFrame:
    """Constructs a crisp outranking relations, based on
    comprehensive concordance C(a, b) and comprehensive
    binary discordance D(a, b) indices.

    :param concordance_comprehensive_table: table with comprehensive concordance indices
    :param discordance_comprehensive_bin_table: table with comprehensive binary
    discordance indices
    :param concordance_cutting_level: concordance majority threshold (cutting level)

    :return: Boolean table the same size as the concordance and discordance tables
    """
    return pd.DataFrame(
        [
            [
                crisp_outranking_Is_marginal(
                    concordance_comprehensive_table.loc[alt_name_a][alt_name_b],
                    discordance_comprehensive_bin_table.loc[alt_name_a][alt_name_b],
                    concordance_cutting_level,
                )
                for alt_name_b in concordance_comprehensive_table.index.values
            ]
            for alt_name_a in concordance_comprehensive_table.index.values
        ],
        index=concordance_comprehensive_table.index,
        columns=concordance_comprehensive_table.index,
    )


def crisp_outranking_coal_marginal(
    concordance_comprehensive: NumericValue,
    discordance_comprehensive: NumericValue,
    concordance_cutting_level: NumericValue,
    discordance_cutting_level: NumericValue,
) -> bool:
    """Constructs a crisp outranking relation, based on
    comprehensive concordance C(a, b) and comprehensive
    discordance D(a, b) indices.

    :param concordance_comprehensive: comprehensive concordance of two alternatives
    :param discordance_comprehensive: comprehensive concordance of two alternatives
    :param concordance_cutting_level: concordance majority threshold (cutting level)
    :param discordance_cutting_level: discordance majority threshold (cutting level)

    :return: ``True`` if a outranks b, ``False`` otherwise
    """
    _check_index_value_interval(concordance_comprehensive, "comprehensive concordance")
    _check_index_value_interval(discordance_comprehensive, "comprehensive discordance")
    _check_index_value_interval(
        concordance_cutting_level, "concordance majority threshold", minimal_val=0.5
    )
    _check_index_value_interval(
        discordance_cutting_level, "discordance majority threshold", include_min=False
    )
    return (
        concordance_comprehensive >= concordance_cutting_level
        and discordance_comprehensive < discordance_cutting_level
    )


def crisp_outranking_coal(
    concordance_comprehensive_table: pd.DataFrame,
    discordance_comprehensive_table: pd.DataFrame,
    concordance_cutting_level: NumericValue,
    discordance_cutting_level: NumericValue,
) -> pd.DataFrame:
    """Constructs a crisp outranking relations, based on
    comprehensive concordance C(a, b) and comprehensive
    discordance D(a, b) indices.

    :param concordance_comprehensive_table: comprehensive concordance table
    :param discordance_comprehensive_table: comprehensive discordance table
    :param concordance_cutting_level: concordance majority threshold (cutting level)
    :param discordance_cutting_level: discordance majority threshold (cutting level)

    :return: Boolean table the same size as the concordance and discordance tables
    """
    return pd.DataFrame(
        [
            [
                crisp_outranking_coal_marginal(
                    concordance_comprehensive_table.loc[alt_name_a][alt_name_b],
                    discordance_comprehensive_table.loc[alt_name_a][alt_name_b],
                    concordance_cutting_level,
                    discordance_cutting_level,
                )
                for alt_name_b in concordance_comprehensive_table.index.values
            ]
            for alt_name_a in concordance_comprehensive_table.index.values
        ],
        index=concordance_comprehensive_table.index,
        columns=concordance_comprehensive_table.index,
    )


def outranking_relation_marginal(
    crisp_outranking_ab: bool,
    crisp_outranking_ba: bool,
) -> Optional[OutrankingRelation]:
    """Aggregates the crisp outranking relations

    :param crisp_outranking_ab: crisp outranking relation of (a, b) alternatives
    :param crisp_outranking_ba: crisp outranking relation of (b, a) alternatives

    :return:
        * None, if b is preferred to a
        * OutrankingRelation enum
    """
    if crisp_outranking_ab and crisp_outranking_ba:
        return OutrankingRelation.INDIFF

    if crisp_outranking_ab and not crisp_outranking_ba:
        return OutrankingRelation.PQ

    if not crisp_outranking_ab and not crisp_outranking_ba:
        return OutrankingRelation.R

    return None


def outranking_relation(
    crisp_outranking_table: pd.DataFrame,
    crisp_outranking_table_profiles: Optional[pd.DataFrame] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Aggregates crisp outranking relations.

    :param crisp_outranking_table: table with crisp relations
    between alternatives or between alternatives and profiles.
    :param crisp_outranking_table_profiles: table with crisp relations
    between profiles and alternatives

    :return: one table with outranking relations between alternatives, if only
    one table was provided, in other case the function will return two
    tables - first one with alternatives - profiles, second one with
    profiles - alternatives comparison.
    """
    if crisp_outranking_table_profiles is not None:
        return pd.DataFrame(
            [
                [
                    outranking_relation_marginal(
                        crisp_outranking_table.loc[alt_name][profile_name],
                        crisp_outranking_table_profiles.loc[profile_name][alt_name],
                    )
                    for profile_name in crisp_outranking_table_profiles.index.values
                ]
                for alt_name in crisp_outranking_table.index.values
            ],
            index=crisp_outranking_table.index,
            columns=crisp_outranking_table_profiles.index,
        ), pd.DataFrame(
            [
                [
                    outranking_relation_marginal(
                        crisp_outranking_table_profiles.loc[profile_name][alt_name],
                        crisp_outranking_table.loc[alt_name][profile_name],
                    )
                    for alt_name in crisp_outranking_table.index.values
                ]
                for profile_name in crisp_outranking_table_profiles.index.values
            ],
            index=crisp_outranking_table_profiles.index,
            columns=crisp_outranking_table.index,
        )

    return pd.DataFrame(
        [
            [
                outranking_relation_marginal(
                    crisp_outranking_table.loc[alt_name_a][alt_name_b],
                    crisp_outranking_table.loc[alt_name_b][alt_name_a],
                )
                for alt_name_b in crisp_outranking_table.index.values
            ]
            for alt_name_a in crisp_outranking_table.index.values
        ],
        index=crisp_outranking_table.index,
        columns=crisp_outranking_table.index,
    )


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
    return (
        1
        if credibility_pair_value_ab > minimal_credibility_index
        and credibility_pair_value_ab
        > credibility_pair_value_ba + linear_function(alpha, credibility_pair_value_ab, beta)
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
    np.fill_diagonal(credibility_matrix.values, 0)
    preference_operator = min if upward_order else max
    remaining_alt_indices = credibility_matrix.index.to_series()
    order = pd.Series([], dtype="float64")
    level: int = 1

    while not remaining_alt_indices.empty:
        preferred_alternatives, minimal_credibility_index = _distillation_process(
            credibility_matrix, remaining_alt_indices, preference_operator, alpha, beta
        )

        if len(preferred_alternatives) > 1:
            preferred_alternatives, _ = _distillation_process(
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


def order_to_outranking_matrix(order: pd.Series) -> pd.DataFrame:
    """Transforms order (upward or downward) to outranking matrix.

    :param order: nested list with order (upward or downward)

    :return: Outranking matrix of given order
    """
    alternatives = order.explode().to_list()
    outranking_matrix = pd.DataFrame(0, index=alternatives, columns=alternatives)

    for position in order:
        outranking_matrix.loc[position, position] = 1
        outranking_matrix.loc[position, alternatives[alternatives.index(position[-1]) + 1:]] = 1

    return outranking_matrix


def final_ranking_matrix(
    descending_order_matrix: pd.DataFrame, ascending_order_matrix: pd.DataFrame
) -> pd.DataFrame:
    """Constructs final partial preorder intersection from downward and upward orders of
    alternatives derived from the descending and ascending distillation procedures, respectively.

    :param descending_order_matrix: outranking matrix from downward order
    :param ascending_order_matrix: outranking matrix from upward order

    :return: Final outranking matrix
    """
    return descending_order_matrix * ascending_order_matrix


def ranks(final_ranking_matrix: pd.DataFrame) -> pd.Series:
    """Constructs ranks of the alternatives in the final preorder.

    :param final_ranking_matrix: outranking matrix from final ranking

    :return: Nested list of ranks
    """
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


def _change_to_series(crisp_outranking_table: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            alt_name_b: [
                alt_name_a
                for alt_name_a in crisp_outranking_table.index
                if crisp_outranking_table.loc[alt_name_b][alt_name_a] != 0
            ]
            for alt_name_b in crisp_outranking_table.index.values
        }
    )


def strongly_connected_components(graph: pd.Series) -> List[List[Any]]:
    index_counter = [0]
    stack, result = [], []
    lowlink, index = {}, {}

    # Function checks if node make with another strongly_connected_component. If so
    # return list of nodes. Otherwise return only this node as a list.
    def _strong_connect(node):
        index[node] = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)

        successors = graph[node]
        for successor in successors:
            if successor not in index:
                _strong_connect(successor)
                lowlink[node] = min(lowlink[node], lowlink[successor])
            elif successor in stack:
                lowlink[node] = min(lowlink[node], index[successor])

        if lowlink[node] == index[node]:
            connected_component = []

            while True:
                successor = stack.pop()
                connected_component.append(successor)
                if successor == node:
                    break
            result.append(connected_component)

    for node in graph.index:
        if node not in index:
            _strong_connect(node)
    return result


def aggregate(graph: pd.Series) -> pd.Series:
    new_graph = graph.copy()
    for vertices in strongly_connected_components(graph):
        if len(vertices) == 1:
            continue
        aggregated = ", ".join(str(v) for v in vertices)
        new_connections = list(
            set([v for key in vertices for v in graph[key] if v not in vertices])
        )
        new_graph = new_graph.drop(labels=vertices)
        for key in new_graph.index.values:
            for vertex in new_graph[key][:]:
                if vertex in vertices:
                    new_graph[key].remove(vertex)
                    if aggregated not in new_graph[key]:
                        new_graph[key].append(aggregated)
        new_graph[aggregated] = new_connections
    for key in new_graph.index.values:
        if key in new_graph[key]:
            new_graph[key].remove(key)
    return new_graph


def find_vertices_without_predecessor(graph: pd.Series) -> List[Any]:
    vertices_with_predecessor = list(set([v for key in graph.index.values for v in graph[key]]))
    return [vertex for vertex in graph.index if vertex not in vertices_with_predecessor]


def find_kernel(crisp_outranking_table: pd.DataFrame) -> List[str]:
    """This function finds a kernel (out1) in a graph
    constructed on the basis of a crisp outranking relation
    :param crisp_outranking_table: table with crisp relations
    between alternatives
    :return: every alternative that is in kernel
    """
    graph = _change_to_series(crisp_outranking_table)
    graph = aggregate(graph)
    not_kernel: List = []
    kernel = find_vertices_without_predecessor(graph)
    for vertex in kernel:
        not_kernel = not_kernel + graph[vertex]
        graph.pop(vertex)
    while len(graph.keys()) != 0:
        vertices = find_vertices_without_predecessor(graph)
        for vertex in vertices:
            if vertex not in not_kernel:
                kernel.append(vertex)
                not_kernel = not_kernel + graph[vertex]
            graph.pop(vertex)
    return kernel


def net_flow_score(crisp_outranking_table: pd.DataFrame) -> pd.Series:
    """This function computes net flow scores for all
    alternatives.
    :param crisp_outranking_table: table with crisp relations
    between alternatives
    :return: net flow scores for all alternatives
    """
    return pd.Series(
        [
            crisp_outranking_table.loc[alt_name].sum() - crisp_outranking_table[alt_name].sum()
            for alt_name in crisp_outranking_table.index.values
        ],
        index=crisp_outranking_table.index,
    ).sort_values(ascending=False)


def median_order(ranks: pd.Series, downward_order: pd.Series, upward_order: pd.Series) -> pd.Series:
    """Constructs median preorder.

    :param ranks: nested list of ranks of the alternatives
    :param downward_order: nested list of downward order
    :param upward_order: nested list of upward order

    :return: Nested list of median preorder
    """
    alternatives = ranks.explode().to_list()
    ranks = transform_series(ranks)
    downward_order = transform_series(downward_order)
    upward_order = transform_series(upward_order)

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
                downwards_difference = downward_order[alt_name_a] - downward_order[alt_name_b]
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
            downwards_difference = downward_order[alt_name_a] - downward_order[alt_name_b]
            upwards_difference = upward_order[alt_name_a] - upward_order[alt_name_b]

            if downwards_difference + upwards_difference > 0:
                level += 1

        final_order[alternatives[initial_order[i]]] = level

    return reverse_transform_series(final_order)


def assign_tri_nb_class(
    crisp_outranking_ap: pd.DataFrame,
    crisp_outranking_pa: pd.DataFrame,
    categories: pd.Series,
    optimistic: bool = True,
) -> pd.Series:
    """_summary_

    :param crisp_outranking_ap: _description_
    :param crisp_outranking_pa: _description_
    :param profiles: _description_
    :param optimistic: _description_
    :return: _description_
    """
    assignment = pd.Series([], dtype=pd.StringDtype(storage=None))
    if not optimistic:
        for alternative in crisp_outranking_ap.index.values:
            for category, profiles in categories.items():
                in_category = False
                for profile in profiles:
                    relation_pa = outranking_relation_marginal(
                        crisp_outranking_pa.loc[profile][alternative],
                        crisp_outranking_ap.loc[alternative][profile],
                    )
                    relation_ap = outranking_relation_marginal(
                        crisp_outranking_ap.loc[alternative][profile],
                        crisp_outranking_pa.loc[profile][alternative],
                    )
                    if relation_ap in {
                        OutrankingRelation.PQ,
                        OutrankingRelation.INDIFF,
                    }:
                        in_category = True
                    if relation_pa == OutrankingRelation.PQ:
                        in_category = False
                        break
                if in_category:
                    assignment[alternative] = category
                    break
            if not in_category:
                assignment[alternative] = categories.index.values[-1]

    if optimistic:
        for alternative in crisp_outranking_ap.index.values:
            current_category = categories.index.values[-1]
            in_category = False
            for category, profiles in categories[-2::-1].items():
                in_category = False
                for profile in profiles:
                    relation_pa = outranking_relation_marginal(
                        crisp_outranking_pa.loc[profile][alternative],
                        crisp_outranking_ap.loc[alternative][profile],
                    )
                    relation_ap = outranking_relation_marginal(
                        crisp_outranking_ap.loc[alternative][profile],
                        crisp_outranking_pa.loc[profile][alternative],
                    )
                    if relation_pa == OutrankingRelation.PQ:
                        in_category = True
                    if relation_ap == OutrankingRelation.PQ:
                        in_category = False
                        break
                if in_category:
                    assignment[alternative] = current_category
                    break
                current_category = category
            if not in_category:
                assignment[alternative] = categories.index.values[0]
    return assignment
