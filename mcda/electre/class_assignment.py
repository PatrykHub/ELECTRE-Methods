from typing import Any, Dict, Union

from mcda.core.aliases import NumericValue
from mcda.electre.outranking import OutrankingRelation, outranking_relation_marginal

import pandas as pd


def assign_tri_rc_class(
    alternatives: Union[Dict[Any, NumericValue], pd.Series],
    categories_rank: pd.Series,
    categories_profiles: pd.Series,
    outranking_ap: pd.DataFrame,
    outranking_pa: pd.DataFrame,
    credibility_ap: pd.DataFrame,
    credibility_pa: pd.DataFrame,
) -> pd.Series:
    """
    :param alternatives:
    :param categories_rank:
    :param categories_profiles:
    :param outranking_ap:
    :param outranking_pa:
    :param credibility_ap:
    :param credibility_pa:
    :return:
    """
    # sorted categories by ranks - ascending (worst to best)
    categories = [i for i in categories_rank.sort_values(ascending=False)]

    # list of profiles according to categories
    profiles = list(sorted(categories_profiles))

    assignments_descending = []
    assignments_ascending = []
    for a in alternatives:
        found_descending = False
        for p in profiles[len(profiles) - 2:: -1]:
            p_next = profiles[profiles.index(p) + 1]
            relation = outranking_relation_marginal(
                outranking_ap.loc[a][p], outranking_pa.loc[p][a]
            )
            if (
                relation == OutrankingRelation.PQ
                and credibility_ap.loc[a][p_next] > credibility_pa.loc[p][a]
            ):
                p_next = categories_profiles[categories_profiles == p_next].index[0]
                category = categories_profiles[p_next]
                assignments_descending.append((a, category))
                found_descending = True
                break

        if not found_descending:
            assignments_descending.append((a, categories[0]))

        found_ascending = False
        for p in profiles[1:]:
            p_prev = profiles[profiles.index(p) - 1]
            relation = outranking_relation_marginal(
                outranking_pa.loc[p][a], outranking_ap.loc[a][p]
            )
            if (
                relation == OutrankingRelation.PQ
                and credibility_pa.loc[p_prev][a] > credibility_ap.loc[a][p]
            ):
                p_prev = categories_profiles[categories_profiles == p_prev].index[0]
                category = categories_profiles[p_prev]
                assignments_ascending.append((a, category))
                found_ascending = True
                break
        if not found_ascending:
            assignments_ascending.append((a, categories[0]))
    assignments = pd.Series()
    for i in zip(assignments_descending, assignments_ascending):
        assignments[i[0][0]] = (i[0][1], i[1][1])
    return assignments


def assign_tri_c_class(
    alternatives: Union[Dict[Any, NumericValue], pd.Series],
    categories_rank: pd.Series,
    categories_profiles: pd.Series,
    outranking_ap: pd.DataFrame,
    outranking_pa: pd.DataFrame,
    credibility_ap: pd.DataFrame,
    credibility_pa: pd.DataFrame,
) -> pd.Series:
    """

    :param alternatives: list of alternatives identifiers
    :param categories_rank: dictionary of categories rankings
    :param categories_profiles: dictionary with characteristic (central) profiles
    :param outranking_ap:
    :param outranking_pa:
    :param credibility_ap:
    :param credibility_pa:
    :return:
    """
    # sorted categories by ranks - ascending (worst to best)
    categories = list(categories_rank.sort_values(ascending=False))

    # list of profiles according to categories
    profiles = list(sorted(categories_profiles))

    assignments_descending = []
    assignments_ascending = []
    for a in alternatives:
        found_descending = False
        for p in profiles[len(profiles) - 2:: -1]:
            p_next = profiles[profiles.index(p) + 1]
            relation = outranking_relation_marginal(
                outranking_ap.loc[a][p], outranking_pa.loc[p][a]
            )
            relation_next = outranking_relation_marginal(
                outranking_ap.loc[a][p_next], outranking_pa.loc[p_next][a]
            )
            if relation == OutrankingRelation.PQ and (
                credibility_ap.loc[a][p_next] > credibility_pa.loc[p][a]
                or credibility_ap.loc[a][p_next] >= credibility_pa.loc[p][a]
                and relation_next == OutrankingRelation.R
            ):
                p_next = categories_profiles[categories_profiles == p_next].index[0]
                category = categories_profiles[p_next]
                assignments_descending.append((a, category))
                found_descending = True
                break
        if not found_descending:
            assignments_descending.append((a, categories[0]))

        found_ascending = False
        for p in profiles[1:]:
            p_prev = profiles[profiles.index(p) - 1]
            relation = outranking_relation_marginal(
                outranking_pa.loc[p][a], outranking_ap.loc[a][p]
            )
            relation_prev = outranking_relation_marginal(
                outranking_ap.loc[a][p_prev], outranking_pa.loc[p_prev][a]
            )
            if relation == OutrankingRelation.PQ and (
                credibility_pa.loc[p_prev][a] > credibility_ap.loc[a][p]
                or credibility_pa.loc[p_prev][a] >= credibility_ap.loc[a][p]
                and relation_prev == OutrankingRelation.R
            ):
                p_prev = categories_profiles[categories_profiles == p_prev].index[0]
                category = categories_profiles[p_prev]
                assignments_ascending.append((a, category))
                found_ascending = True
                break
        if not found_ascending:
            assignments_ascending.append((a, categories[0]))
    assignments = pd.Series()
    for i in zip(assignments_descending, assignments_ascending):
        assignments[i[0][0]] = (i[0][1], i[1][1])
    return assignments


def assign_tri_class(
    alternatives: Union[Dict[Any, NumericValue], pd.Series],
    categories_rank: pd.Series,
    categories_profiles: pd.Series,
    crisp_outranking_ap: pd.DataFrame,
    crisp_outranking_pa: pd.DataFrame,
) -> pd.Series:
    """
    :param alternatives: list of alternatives identifiers
    :param categories_rank: dictionary of categories rankings
    :param categories_profiles: dictionary with boundary profiles
    :param crisp_outranking_ap: dataframe containing crisp outranking values
    :param crisp_outranking_pa:
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
                crisp_outranking_ap.loc[alternative][profile],
                crisp_outranking_pa.loc[profile][alternative],
            )
            if relation in (OutrankingRelation.INDIFF, OutrankingRelation.PQ):
                pessimistic_idx = i + 1
                break

        # Optimistic assignment
        optimistic_idx = len(categories_profiles)
        for i, profile in list(enumerate(categories_profiles.index)):
            relation = outranking_relation_marginal(
                crisp_outranking_pa.loc[profile][alternative],
                crisp_outranking_ap.loc[alternative][profile],
            )
            if relation == OutrankingRelation.PQ:
                optimistic_idx = i
                break

        assignment[alternative] = (
            categories[pessimistic_idx],
            categories[optimistic_idx],
        )
    return assignment
