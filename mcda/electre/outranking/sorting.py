"""This module implements methods to explore outranking relations with sorting problems."""
import pandas as pd

from mcda.electre.outranking import OutrankingRelation, outranking_relation_marginal


def assign_tri_b_class(
    crisp_outranking_alt_prof: pd.DataFrame,
    crisp_outranking_prof_alt: pd.DataFrame,
    boundary_profiles: pd.Series,
) -> pd.Series:
    """Assigns each element of the alternatives set to appropriate class,
    based on the crisp outranking relations between alternatives and boundary profiles.

    :param crisp_outranking_alt_prof: crisp outranking relation DataFrame alternatives-profiles
    :param crisp_outranking_prof_alt: crisp outranking relation DataFrame profiles-alternatives
    :param boundary_profiles: profiles which separate the classes

    :return: Series of pairs with the pessimistic and optimistic assignment to the classes
    """
    assignment = pd.Series([], dtype=pd.StringDtype(storage=None))

    for alternative in crisp_outranking_alt_prof.index.values:
        # Pessimistic assignment
        pessimistic_idx = 0
        for i, profile in list(enumerate(boundary_profiles.values))[1::-1]:
            relation = outranking_relation_marginal(
                crisp_outranking_alt_prof.loc[alternative, profile],
                crisp_outranking_prof_alt.loc[profile, alternative],
            )
            if relation in (OutrankingRelation.INDIFF, OutrankingRelation.PQ):
                pessimistic_idx = i + 1
                break

        # Optimistic assignment
        optimistic_idx = len(boundary_profiles) - 1
        for i, profile in list(enumerate(boundary_profiles.values))[:-1]:
            relation = outranking_relation_marginal(
                crisp_outranking_prof_alt.loc[profile, alternative],
                crisp_outranking_alt_prof.loc[alternative, profile],
            )
            if relation == OutrankingRelation.PQ:
                optimistic_idx = i
                break

        assignment[alternative] = (
            boundary_profiles.index.values[pessimistic_idx],
            boundary_profiles.index.values[optimistic_idx],
        )
    return assignment


def assign_tri_nb_class(
    crisp_outranking_alt_prof: pd.DataFrame,
    crisp_outranking_prof_alt: pd.DataFrame,
    boundary_profiles: pd.Series,
) -> pd.Series:
    """Assigns each element of the alternatives set to appropriate class,
    based on the crisp outranking relations between alternatives and boundary profiles.

    :param crisp_outranking_alt_prof: crisp outranking relation DataFrame alternatives-profiles
    :param crisp_outranking_prof_alt: crisp outranking relation DataFrame profiles-alternatives
    :param boundary_profiles: profiles which separate the classes

    :return: Series of pairs with the pessimistic and optimistic assignment to the classes
    """
    assignment = pd.Series([], dtype=pd.StringDtype(storage=None))
    for alternative in crisp_outranking_alt_prof.index.values:
        for category, profiles in boundary_profiles.items():
            in_category = False
            for profile in profiles:
                relation_pa = outranking_relation_marginal(
                    crisp_outranking_prof_alt.loc[profile][alternative],
                    crisp_outranking_alt_prof.loc[alternative][profile],
                )
                relation_ap = outranking_relation_marginal(
                    crisp_outranking_alt_prof.loc[alternative][profile],
                    crisp_outranking_prof_alt.loc[profile][alternative],
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
            assignment_pesimistic = boundary_profiles.index.values[-1]
        current_category = boundary_profiles.index.values[-1]
        in_category = False
        for category, profiles in boundary_profiles[-2::-1].items():
            in_category = False
            for profile in profiles:
                relation_pa = outranking_relation_marginal(
                    crisp_outranking_prof_alt.loc[profile][alternative],
                    crisp_outranking_alt_prof.loc[alternative][profile],
                )
                relation_ap = outranking_relation_marginal(
                    crisp_outranking_alt_prof.loc[alternative][profile],
                    crisp_outranking_prof_alt.loc[profile][alternative],
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
            assignment_optimistic = boundary_profiles.index.values[0]
        assignment[alternative] = (assignment_pesimistic, assignment_optimistic)
    return assignment


def assign_tri_c_class(
    crisp_outranking_alt_prof: pd.DataFrame,
    crisp_outranking_prof_alt: pd.DataFrame,
    credibility_alt_prof: pd.DataFrame,
    credibility_prof_alt: pd.DataFrame,
    characteristic_profiles: pd.Series,
) -> pd.Series:
    """Implements the descending and ascending assignment rules for set of alternatives,
    based on crisp outarnking relation, credibility tables and characteristic profiles.

    :param crisp_outranking_alt_prof: crisp outranking relation DataFrame alternatives-profiles
    :param crisp_outranking_prof_alt: crisp outranking relation DataFrame profiles-alternatives
    :param credibility_alt_prof: _description_
    :param credibility_prof_alt: _description_
    :param characteristic_profiles: _description_

    :return: _description_
    """
    assignment = pd.Series([], dtype=pd.StringDtype(storage=None))

    assignments_descending = []
    assignments_ascending = []
    for alternative in crisp_outranking_alt_prof.index.values:
        found_descending = False
        for i, profile in enumerate(
            characteristic_profiles[len(characteristic_profiles) - 2:: -1]
        ):
            p_next = characteristic_profiles.iloc[len(characteristic_profiles) - i - 1]
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
                category = characteristic_profiles[characteristic_profiles == p_next].index[0]
                assignments_descending.append((alternative, category))
                found_descending = True
                break
        if not found_descending:
            assignments_descending.append((alternative, characteristic_profiles.index[0]))

        found_ascending = False
        for i, profile in enumerate(characteristic_profiles[1:]):
            p_prev = characteristic_profiles.iloc[i]
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
                category = characteristic_profiles[characteristic_profiles == p_prev].index[0]
                assignments_ascending.append((alternative, category))
                found_ascending = True
                break
        if not found_ascending:
            assignments_ascending.append((alternative, characteristic_profiles.index[-1]))
    for zipped in zip(assignments_descending, assignments_ascending):
        assignment[zipped[0][0]] = (zipped[0][1], zipped[1][1])
    return assignment


def assign_tri_rc_class(
    crisp_outranking_alt_prof: pd.DataFrame,
    crisp_outranking_prof_alt: pd.DataFrame,
    credibility_alt_prof: pd.DataFrame,
    credibility_prof_alt: pd.DataFrame,
    characteristic_profiles: pd.Series,
) -> pd.Series:
    """_summary_

    :param crisp_outranking_alt_prof: crisp outranking relation DataFrame alternatives-profiles
    :param crisp_outranking_prof_alt: crisp outranking relation DataFrame profiles-alternatives
    :param credibility_alt_prof: _description_
    :param credibility_prof_alt: _description_
    :param characteristic_profiles: _description_
    
    :return: _description_
    """
    assignment = pd.Series([], dtype=pd.StringDtype(storage=None))

    assignments_descending = []
    assignments_ascending = []
    for alternative in crisp_outranking_alt_prof.index.values:
        found_descending = False
        for i, profile in enumerate(
            characteristic_profiles[len(characteristic_profiles) - 2:: -1]
        ):
            p_next = characteristic_profiles.iloc[len(characteristic_profiles) - i - 1]
            relation = outranking_relation_marginal(
                crisp_outranking_alt_prof.loc[alternative][profile],
                crisp_outranking_prof_alt.loc[profile][alternative],
            )
            if (
                relation == OutrankingRelation.PQ
                and credibility_alt_prof.loc[alternative][p_next]
                > credibility_prof_alt.loc[profile][alternative]
            ):
                category = characteristic_profiles[characteristic_profiles == p_next].index[0]
                assignments_descending.append((alternative, category))
                found_descending = True
                break

        if not found_descending:
            assignments_descending.append((alternative, characteristic_profiles.index[0]))

        found_ascending = False
        for i, profile in enumerate(characteristic_profiles[1:]):
            p_prev = characteristic_profiles.iloc[i]
            relation = outranking_relation_marginal(
                crisp_outranking_prof_alt.loc[profile][alternative],
                crisp_outranking_alt_prof.loc[alternative][profile],
            )
            if (
                relation == OutrankingRelation.PQ
                and credibility_prof_alt.loc[p_prev][alternative]
                > credibility_alt_prof.loc[alternative][profile]
            ):
                category = characteristic_profiles[characteristic_profiles == p_prev].index[0]
                assignments_ascending.append((alternative, category))
                found_ascending = True
                break
        if not found_ascending:
            assignments_ascending.append((alternative, characteristic_profiles.index[-1]))
    for zipped in zip(assignments_descending, assignments_ascending):
        assignment[zipped[0][0]] = (zipped[0][1], zipped[1][1])
    return assignment
