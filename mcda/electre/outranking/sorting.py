"""This module implements modules to explore outranking with sorting methods."""
import pandas as pd

from mcda.electre.outranking import OutrankingRelation, outranking_relation_marginal


def assign_tri_b_class(
    categories_profiles: pd.Series,
    crisp_outranking_alt_prof: pd.DataFrame,
    crisp_outranking_prof_alt: pd.DataFrame,
) -> pd.Series:
    """
    This function assigns alternatives to classes according to the outranking.
    :param categories_profiles: dictionary with boundary profiles
    :param crisp_outranking_alt_prof: DataFrame containing crisp outranking values
    :param crisp_outranking_prof_alt:
    :return:
    """
    assignment = pd.Series([], dtype=pd.StringDtype(storage=None))

    for alternative in crisp_outranking_alt_prof.index.values:
        # Pessimistic assignment
        pessimistic_idx = 0
        for i, profile in list(enumerate(categories_profiles.values))[1::-1]:
            relation = outranking_relation_marginal(
                crisp_outranking_alt_prof.loc[alternative, profile],
                crisp_outranking_prof_alt.loc[profile, alternative],
            )
            if relation in (OutrankingRelation.INDIFF, OutrankingRelation.PQ):
                pessimistic_idx = i + 1
                break

        # Optimistic assignment
        optimistic_idx = len(categories_profiles) - 1
        for i, profile in list(enumerate(categories_profiles.values))[:-1]:
            relation = outranking_relation_marginal(
                crisp_outranking_prof_alt.loc[profile, alternative],
                crisp_outranking_alt_prof.loc[alternative, profile],
            )
            if relation == OutrankingRelation.PQ:
                optimistic_idx = i
                break

        assignment[alternative] = (
            categories_profiles.index.values[pessimistic_idx],
            categories_profiles.index.values[optimistic_idx],
        )
    return assignment


def assign_tri_nb_class(
    crisp_outranking_ap: pd.DataFrame,
    crisp_outranking_pa: pd.DataFrame,
    categories: pd.Series,
) -> pd.Series:
    """_summary_

    :param crisp_outranking_ap: _description_
    :param crisp_outranking_pa: _description_
    :param profiles: _description_
    :return: _description_
    """
    assignment = pd.Series([], dtype=pd.StringDtype(storage=None))
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
                assignment_pesimistic = category
                break
        if not in_category:
            assignment_pesimistic = categories.index.values[-1]
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
                assignment_optimistic = current_category
                break
            current_category = category
        if not in_category:
            assignment_optimistic = categories.index.values[0]
        assignment[alternative] = (assignment_pesimistic, assignment_optimistic)
    return assignment


def assign_tri_c_class(
    categories_profiles: pd.Series,
    crisp_outranking_alt_prof: pd.DataFrame,
    crisp_outranking_prof_alt: pd.DataFrame,
    credibility_alt_prof: pd.DataFrame,
    credibility_prof_alt: pd.DataFrame,
) -> pd.Series:
    """
    :param categories_profiles: dictionary with characteristic (central) profiles
    :param crisp_outranking_alt_prof:
    :param crisp_outranking_prof_alt:
    :param credibility_alt_prof:
    :param credibility_prof_alt:
    :return:
    """

    assignment = pd.Series([], dtype=pd.StringDtype(storage=None))

    assignments_descending = []
    assignments_ascending = []
    for alternative in crisp_outranking_alt_prof.index.values:
        found_descending = False
        for i, profile in enumerate(
            categories_profiles[len(categories_profiles) - 2:: -1]
        ):
            p_next = categories_profiles.iloc[len(categories_profiles) - i - 1]
            relation = outranking_relation_marginal(
                crisp_outranking_alt_prof.loc[alternative][profile],
                crisp_outranking_prof_alt.loc[profile][alternative],
            )
            relation_next = outranking_relation_marginal(
                crisp_outranking_alt_prof.loc[alternative][p_next],
                crisp_outranking_prof_alt.loc[p_next][alternative],
            )
            if relation == OutrankingRelation.PQ and (
                credibility_alt_prof.loc[alternative][p_next]
                > credibility_prof_alt.loc[profile][alternative]
                or credibility_alt_prof.loc[alternative][p_next]
                >= credibility_prof_alt.loc[profile][alternative]
                and relation_next == OutrankingRelation.R
            ):
                category = categories_profiles[categories_profiles == p_next].index[0]
                assignments_descending.append((alternative, category))
                found_descending = True
                break
        if not found_descending:
            assignments_descending.append((alternative, categories_profiles.index[0]))

        found_ascending = False
        for i, profile in enumerate(categories_profiles[1:]):
            p_prev = categories_profiles.iloc[i]
            relation = outranking_relation_marginal(
                crisp_outranking_prof_alt.loc[profile][alternative],
                crisp_outranking_alt_prof.loc[alternative][profile],
            )
            relation_prev = outranking_relation_marginal(
                crisp_outranking_alt_prof.loc[alternative][p_prev],
                crisp_outranking_prof_alt.loc[p_prev][alternative],
            )
            if relation == OutrankingRelation.PQ and (
                credibility_prof_alt.loc[p_prev][alternative]
                > credibility_alt_prof.loc[alternative][profile]
                or credibility_prof_alt.loc[p_prev][alternative]
                >= credibility_alt_prof.loc[alternative][profile]
                and relation_prev == OutrankingRelation.R
            ):
                category = categories_profiles[categories_profiles == p_prev].index[0]
                assignments_ascending.append((alternative, category))
                found_ascending = True
                break
        if not found_ascending:
            assignments_ascending.append((alternative, categories_profiles.index[-1]))
    for zipped in zip(assignments_descending, assignments_ascending):
        assignment[zipped[0][0]] = (zipped[0][1], zipped[1][1])
    return assignment


def assign_tri_rc_class(
    categories_profiles: pd.Series,
    crisp_outranking_alt_prof: pd.DataFrame,
    crisp_outranking_prof_alt: pd.DataFrame,
    credibility_alt_prof: pd.DataFrame,
    credibility_prof_alt: pd.DataFrame,
) -> pd.Series:
    """

    :param categories_profiles:
    :param crisp_outranking_alt_prof:
    :param crisp_outranking_prof_alt:
    :param credibility_alt_prof:
    :param credibility_prof_alt:
    :return:
    """
    assignment = pd.Series([], dtype=pd.StringDtype(storage=None))

    assignments_descending = []
    assignments_ascending = []
    for alternative in crisp_outranking_alt_prof.index.values:
        found_descending = False
        for i, profile in enumerate(
            categories_profiles[len(categories_profiles) - 2:: -1]
        ):
            p_next = categories_profiles.iloc[len(categories_profiles) - i - 1]
            relation = outranking_relation_marginal(
                crisp_outranking_alt_prof.loc[alternative][profile],
                crisp_outranking_prof_alt.loc[profile][alternative],
            )
            if (
                relation == OutrankingRelation.PQ
                and credibility_alt_prof.loc[alternative][p_next]
                > credibility_prof_alt.loc[profile][alternative]
            ):
                category = categories_profiles[categories_profiles == p_next].index[0]
                assignments_descending.append((alternative, category))
                found_descending = True
                break

        if not found_descending:
            assignments_descending.append((alternative, categories_profiles.index[0]))

        found_ascending = False
        for i, profile in enumerate(categories_profiles[1:]):
            p_prev = categories_profiles.iloc[i]
            relation = outranking_relation_marginal(
                crisp_outranking_prof_alt.loc[profile][alternative],
                crisp_outranking_alt_prof.loc[alternative][profile],
            )
            if (
                relation == OutrankingRelation.PQ
                and credibility_prof_alt.loc[p_prev][alternative]
                > credibility_alt_prof.loc[alternative][profile]
            ):
                category = categories_profiles[categories_profiles == p_prev].index[0]
                assignments_ascending.append((alternative, category))
                found_ascending = True
                break
        if not found_ascending:
            assignments_ascending.append((alternative, categories_profiles.index[-1]))
    for zipped in zip(assignments_descending, assignments_ascending):
        assignment[zipped[0][0]] = (zipped[0][1], zipped[1][1])
    return assignment
