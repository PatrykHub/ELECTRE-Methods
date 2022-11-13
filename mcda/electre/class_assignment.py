from typing import Any, Dict, Union

from ..core.aliases import NumericValue
from ..electre.outranking import OutrankingRelation, outranking_relation_marginal

import pandas as pd


def assign_tri_rc_class(
    alternatives: Union[Dict[Any, NumericValue], pd.Series],
    categories_rank: pd.Series,
    categories_profiles: pd.Series,
    outranking: pd.DataFrame,
    credibility: pd.DataFrame,
) -> pd.Series:
    """

    :param alternatives:
    :param categories_rank:
    :param categories_profiles:
    :param outranking:
    :param credibility:
    :return:
    """
    # sorted categories by ranks - ascending (worst to best)
    categories = [i for i in categories_rank.sort_values(ascending=False)]

    # list of profiles according to categories
    profiles = list(sorted(categories_profiles, key=lambda x: categories.index(x)))

    assignments_descending = []
    assignments_ascending = []
    for a in alternatives:
        found_descending = False
        for p in profiles[len(profiles) - 2 :: -1]:
            p_next = profiles[profiles.index(p) + 1]
            relation = outranking_relation_marginal(
                outranking.loc[a][p], outranking.loc[p][a]
            )
            if (
                relation == OutrankingRelation.PQ
                and credibility.loc[a][p_next] > credibility.loc[p][a]
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
            relation = outranking_relation_marginal(
                outranking.loc[p][a], outranking.loc[a][p]
            )
            if (
                relation == OutrankingRelation.PQ
                and credibility.loc[p_prev][a] > credibility.loc[a][p]
            ):
                category = categories_profiles.get(p_prev)
                assignments_ascending.append((a, category))
                found_ascending = True
                break
        if not found_ascending:
            assignments_ascending.append((a, categories[-1]))
    assignments = pd.Series()
    for i in zip(assignments_descending, assignments_ascending):
        assignments[i[0][0]] = (i[0][1], i[1][1])
    return assignments


def assign_tri_c_class(
    alternatives: Union[Dict[Any, NumericValue], pd.Series],
    categories_rank: pd.Series,
    categories_profiles: pd.Series,
    outranking: pd.DataFrame,
    credibility: pd.DataFrame,
) -> pd.Series:
    """

    :param alternatives: list of alternatives identifiers
    :param categories_rank: dictionary of categories rankings
    :param categories_profiles: dictionary with characteristic profiles
    :param outranking: dataframe containing outranking values
    :param credibility: matrix of credibility values
    :return:
    """
    # sorted categories by ranks - ascending (worst to best)
    categories = list(categories_rank.sort_values(ascending=False))

    # list of profiles according to categories
    profiles = list(sorted(categories_profiles, key=lambda x: categories.index(x)))
    assignments_descending = []
    assignments_ascending = []
    for a in alternatives:
        found_descending = False
        for p in profiles[len(profiles) - 2 :: -1]:
            p_next = profiles[profiles.index(p) + 1]
            relation = outranking_relation_marginal(
                outranking.loc[a][p], outranking.loc[p][a]
            )
            relation_next = outranking_relation_marginal(
                outranking.loc[a][p_next], outranking.loc[p_next][a]
            )
            if relation == OutrankingRelation.PQ and (
                credibility.loc[a][p_next] > credibility.loc[p][a]
                or credibility.loc[a][p_next] >= credibility.loc[p][a]
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
            relation = outranking_relation_marginal(
                outranking.loc[p][a], outranking.loc[a][p]
            )
            relation_prev = outranking_relation_marginal(
                outranking.loc[a][p_prev], outranking.loc[p_prev][a]
            )
            if relation == OutrankingRelation.PQ and (
                credibility.loc[p_prev][a] > credibility.loc[a][p]
                or credibility.loc[p_prev][a] >= credibility.loc[a][p]
                and relation_prev == OutrankingRelation.R
            ):
                category = categories_profiles.get(p_prev)
                assignments_ascending.append((a, category))
                found_ascending = True
                break
        if not found_ascending:
            assignments_ascending.append((a, categories[-1]))

    assignments = pd.Series()
    for i in zip(assignments_descending, assignments_ascending):
        assignments[i[0][0]] = (i[0][1], i[1][1])
    return assignments


def assign_tri_class(
    alternatives: Union[Dict[Any, NumericValue], pd.Series],
    categories_rank: pd.Series,
    categories_profiles: pd.Series,
    crisp_outranking: pd.DataFrame,
) -> pd.Series:
    """
    :param alternatives: list of alternatives identifiers
    :param categories_rank: dictionary of categories rankings
    :param categories_profiles: dictionary with boundary profiles
    :param crisp_outranking: dataframe containing crisp outranking values
    :return:
    """
    # Initiate categories to assign
    categories = list(categories_rank.sort_index(ascending=False))

    assignment = pd.Series()
    for alternative in alternatives:
        # Pessimistic assignment
        pessimistic_idx = 0
        for i, profile in list(enumerate(categories_profiles.index))[::-1]:
            relation = outranking_relation_marginal(
                crisp_outranking.loc[alternative][profile],
                crisp_outranking.loc[profile][alternative],
            )
            if relation in (OutrankingRelation.INDIFF, OutrankingRelation.PQ):
                pessimistic_idx = i + 1
                break

        # Optimistic assignment
        optimistic_idx = len(categories_profiles)
        for i, profile in list(enumerate(categories_profiles.index)):
            relation = outranking_relation_marginal(
                crisp_outranking.loc[profile][alternative],
                crisp_outranking.loc[alternative][profile],
            )
            if relation == OutrankingRelation.PQ:
                optimistic_idx = i
                break

        assignment[alternative] = (
            categories[pessimistic_idx],
            categories[optimistic_idx],
        )
    return assignment
