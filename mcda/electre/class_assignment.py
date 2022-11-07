from typing import Any, Dict, List, Union

from ..core.aliases import NumericValue
from ..electre.outranking import OutrankingRelation

import pandas as pd


def get_relation_type(
    x: Union[int, str], y: Union[int, str], outranking: pd.DataFrame
) -> OutrankingRelation:
    """
    Assigns type of relation according to credibility
    :param x: index of row
    :param y: index of column
    :param outranking: dataframe containing outranking values
    :return:
    """
    if outranking[x][y] and outranking[y][x]:
        relation = OutrankingRelation.INDIFF
    elif outranking[x][y] and not outranking[y][x]:
        relation = OutrankingRelation.PQ
    elif not outranking[x][y] and not outranking[y][x]:
        relation = OutrankingRelation.R
    else:
        relation = OutrankingRelation.R

    return relation


def assign_tri_c_class(
    alternatives: Union[Dict[Any, NumericValue], pd.Series],
    categories_rank: pd.Series,
    categories_profiles: pd.Series,
    outranking: pd.DataFrame,
    credibility: pd.DataFrame
):
    """

    :param alternatives: list of alternatives identifiers
    :param categories_rank: dictionary of categories rankings
    :param categories_profiles: dictionary with characteristic profiles
    :param outranking: dataframe containing outranking values
    :param credibility: matrix of credibility values
    :return:
    """
    # sorted categories by ranks - ascending (worst to best)
    categories = [
        i for i in categories_rank.sort_values(ascending=False)
    ]

    # list of profiles according to categories
    profiles = [
        i
        for i in sorted(categories_profiles, key=lambda x: categories.index(x))
    ]
    assignments_descending = []
    assignments_ascending = []
    for a in alternatives:
        found_descending = False
        for p in profiles[len(profiles) - 2:: -1]:
            p_next = profiles[profiles.index(p) + 1]
            relation = get_relation_type(a, p, outranking)
            relation_next = get_relation_type(a, p_next, outranking)
            if relation == OutrankingRelation.PQ and (
                credibility[a][p_next] > credibility[p][a]
                or credibility[a][p_next] >= credibility[p][a]
                and relation_next == OutrankingRelation.R
            ):
                category = categories_profiles.get(p_next)
                assignments_descending.append((a, category))
                found_descending = True
                break
        if not found_descending:
            assignments_descending.append((a, categories[0]))

        found_ascending = False
        for p in profiles[1:]:
            p_prev = profiles[profiles.index(p) - 1]
            relation = get_relation_type(p, a, outranking)
            relation_prev = get_relation_type(a, p_prev, outranking)
            if relation == OutrankingRelation.PQ and (
                credibility[p_prev][a] > credibility[a][p]
                or credibility[p_prev][a] >= credibility[a][p]
                and relation_prev == OutrankingRelation.R
            ):
                category = categories_profiles.get(p_prev)
                assignments_ascending.append((a, category))
                found_ascending = True
                break
        if not found_ascending:
            assignments_ascending.append((a, categories[-1]))
    assignments = {}
    for i in zip(assignments_descending, assignments_ascending):
        assignments[i[0][0]] = (i[0][1], i[1][1])
    return assignments


def assign_tri_b_class(
    alternatives: Union[Dict[Any, NumericValue], pd.Series],
    categories_rank: pd.Series,
    categories_profiles: pd.Series,
    crisp_outranking: pd.DataFrame
):
    """
    :param alternatives: list of alternatives identifiers
    :param categories_rank: dictionary of categories rankings
    :param categories_profiles: dictionary with boundary profiles
    :param crisp_outranking: dataframe containing crisp outranking values
    :return:
    """
    # Initiate categories to assign
    categories = [
        i for i in categories_rank.sort_values(ascending=False)
    ]
    assignment = {}
    for alternative in alternatives:
        # Pessimistic assignment
        pessimistic_idx = 0
        for i, profile in list(enumerate(categories_profiles))[::-1]:
            relation = get_relation_type(alternative, profile, crisp_outranking)
            if relation in (OutrankingRelation.INDIFF, OutrankingRelation.PQ):
                pessimistic_idx = i + 1
                break

        # Optimistic assignment
        optimistic_idx = len(categories_profiles)
        for i, profile in enumerate(categories_profiles):
            relation = get_relation_type(profile, alternative, crisp_outranking)
            if relation == OutrankingRelation.PQ:
                optimistic_idx = i
                break

        assignment[alternative] = (
            categories[pessimistic_idx],
            categories[optimistic_idx],
        )
    return assignment
