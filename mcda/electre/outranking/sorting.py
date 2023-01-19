"""This module implements methods to explore outranking relations with sorting problems.

Implementation based on:
    * assign_tri_b_class: :cite:p:`Mousseau00`,
    * assign_tri_nb_class: :cite:p:`Fernandez17`,
    * assign_tri_c_class: :cite:p:`Almeida10`,
    * assign_tri_nc_class: :cite:p:`Almeida12`,
    * assign_tri_rc_class: :cite:p:`Rezaei17`.
"""
from typing import Iterable

import pandas as pd

from .. import exceptions
from .._validation import (
    _check_index_value_binary,
    _check_index_value_interval,
    _check_tables_ap,
    _unique_names,
)
from .crisp_outranking import (
    OutrankingRelation,
    crisp_cut,
    outranking_relation_marginal,
)


def assign_tri_b_class(
    crisp_outranking_alt_prof: pd.DataFrame,
    crisp_outranking_prof_alt: pd.DataFrame,
    boundary_profiles: pd.Series,
) -> pd.Series:
    """Assigns each element of the alternatives set to appropriate class,
    based on the crisp outranking relations between alternatives and boundary profiles.

    :param crisp_outranking_alt_prof: crisp outranking relation table alternatives-profiles
    :param crisp_outranking_prof_alt: crisp outranking relation table profiles-alternatives
    :param boundary_profiles: profiles which separate the classes in ascending order

    :return: `pandas.Series` of pairs with the pessimistic and optimistic assignment to the classes
    """
    assignment = pd.Series([], dtype=pd.StringDtype(storage=None))

    _check_tables_ap(
        crisp_outranking_alt_prof,
        crisp_outranking_prof_alt,
        index_validation_function=_check_index_value_binary,
        name="crisp outranking",
    )
    try:
        _unique_names(boundary_profiles.keys(), names_type="categories")
    except AttributeError as exc:
        raise TypeError(
            f"Wrong boundary_profiles type. Expected {pd.Series.__name__}, "
            f"but got {type(boundary_profiles).__name__} instead."
        ) from exc

    for alternative in crisp_outranking_alt_prof.index.values:
        # Pessimistic assignment
        pessimistic_idx = 0
        try:
            for i, profile in list(enumerate(boundary_profiles.values))[1::-1]:
                relation = outranking_relation_marginal(
                    crisp_outranking_alt_prof.loc[alternative, profile],
                    crisp_outranking_prof_alt.loc[profile, alternative],
                )
                if relation in (OutrankingRelation.INDIFF, OutrankingRelation.PQ):
                    pessimistic_idx = i + 1
                    break
        except KeyError as exc:
            raise exceptions.SortingError(
                "Boundary profiles list contains profile which "
                f"does not exists in crisp outranking tables: {profile}"
            ) from exc

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
    Unlike in `assign_tri_b_class` module, it is possible to define many boundary profiles
    dividing alternatives between every two classes.

    :param crisp_outranking_alt_prof: crisp outranking relation table alternatives-profiles
    :param crisp_outranking_prof_alt: crisp outranking relation table profiles-alternatives
    :param boundary_profiles: profiles which separate the classes in ascending order

    :return: `pandas.Series` of pairs with the pessimistic and optimistic assignment to the classes
    """
    assignment = pd.Series([], dtype=pd.StringDtype(storage=None))
    _check_tables_ap(
        crisp_outranking_alt_prof,
        crisp_outranking_prof_alt,
        index_validation_function=_check_index_value_binary,
        name="crisp outranking",
    )
    try:
        _unique_names(boundary_profiles.keys(), names_type="categories")
    except AttributeError as exc:
        raise TypeError(
            f"Wrong boundary_profiles type. Expected {pd.Series.__name__}, "
            f"but got {type(boundary_profiles).__name__} instead."
        ) from exc

    for alternative in crisp_outranking_alt_prof.index.values:
        try:
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
                    if relation_pa == OutrankingRelation.PQ:
                        in_category = True
                    if relation_ap == OutrankingRelation.PQ:
                        in_category = False
                        break
                if in_category:
                    assignment_optimistic = category
                    break
        except KeyError as exc:
            raise exceptions.SortingError(
                "Boundary profiles list contains profile which "
                f"does not exists in crisp outranking tables: {profile}"
            ) from exc
        except TypeError as exc:
            if not isinstance(profiles, Iterable):
                exc.args = (
                    "Value of the boundary profiles series must be an iterable, "
                    f"but got {type(profiles).__name__} instead.",
                )
            raise

        if not in_category:
            assignment_optimistic = boundary_profiles.index.values[-1]
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
                if relation_ap in {
                    OutrankingRelation.PQ,
                    OutrankingRelation.INDIFF,
                }:
                    in_category = True
                if relation_pa == OutrankingRelation.PQ:
                    in_category = False
                    break
            if in_category:
                assignment_pesimistic = current_category
                break
            current_category = category
        if not in_category:
            assignment_pesimistic = boundary_profiles.index.values[0]
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
    based on crisp outranking relation, credibility tables and characteristic profiles.

    :param crisp_outranking_alt_prof: crisp outranking relation table alternatives-profiles
    :param crisp_outranking_prof_alt: crisp outranking relation table profiles-alternatives
    :param credibility_alt_prof: credibility table alternatives-profiles
    :param credibility_prof_alt: credibility table profiles-alternatives
    :param characteristic_profiles: profiles which characterize classes in ascending order

    :return: `pandas.Series` of pairs with the descending and ascending assignment to the classes
    """
    assignment = pd.Series([], dtype=pd.StringDtype(storage=None))

    _check_tables_ap(
        crisp_outranking_alt_prof,
        crisp_outranking_prof_alt,
        index_validation_function=_check_index_value_binary,
        name="crisp outranking",
    )
    _check_tables_ap(
        credibility_alt_prof,
        credibility_prof_alt,
        index_validation_function=_check_index_value_interval,
        name="credibility",
    )
    try:
        _unique_names(characteristic_profiles.keys(), names_type="categories")
    except AttributeError as exc:
        raise TypeError(
            f"Wrong characteristic_profiles type. Expected {pd.Series.__name__}, "
            f"but got {type(characteristic_profiles).__name__} instead."
        ) from exc

    assignments_descending = []
    assignments_ascending = []
    for alternative in crisp_outranking_alt_prof.index.values:
        found_descending = False
        try:
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
        except KeyError as exc:
            raise exceptions.SortingError(
                "Characteristic profiles list contains profile which "
                "does not exists in crisp outranking / credibility tables."
            ) from exc

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


def assign_tri_nc_class(
    credibility_alt_prof: pd.DataFrame,
    credibility_prof_alt: pd.DataFrame,
    characteristic_profiles: pd.Series,
) -> pd.Series:
    """Implements the descending and ascending assignment rules for set of alternatives,
    based on credibility tables and a set of characteristic profiles.

    :param credibility_alt_prof: credibility table alternatives-profiles
    :param credibility_prof_alt: credibility table profiles-alternatives
    :param characteristic_profiles: profiles which characterize classes in ascending order

    :return: `pandas.Series` of pairs with the descending and ascending
    assignment to the classes
    """
    _check_tables_ap(
        credibility_alt_prof,
        credibility_prof_alt,
        index_validation_function=_check_index_value_interval,
        name="credibility",
    )
    try:
        _unique_names(characteristic_profiles.keys(), names_type="categories")
    except AttributeError as exc:
        raise TypeError(
            f"Wrong characteristic_profiles type. Expected {pd.Series.__name__}, "
            f"but got {type(characteristic_profiles).__name__} instead."
        ) from exc

    credibility_acc_alt_prof = pd.DataFrame(
        [
            [0.0 for _ in characteristic_profiles.index.values]
            for _ in credibility_alt_prof.index.values
        ],
        index=credibility_alt_prof.index.values,
        columns=characteristic_profiles.index.values,
    )
    credibility_acc_prof_alt = pd.DataFrame(
        [
            [0.0 for _ in credibility_prof_alt.columns.values]
            for _ in characteristic_profiles.index.values
        ],
        index=characteristic_profiles.index.values,
        columns=credibility_prof_alt.columns.values,
    )
    for alternative in credibility_alt_prof.index.values:
        try:
            for i, profile in enumerate(characteristic_profiles):
                cat_idx = characteristic_profiles.index.values[i]
                for category in profile:
                    if (
                        credibility_alt_prof.loc[alternative][category]
                        > credibility_acc_alt_prof.loc[alternative][cat_idx]
                    ):
                        credibility_acc_alt_prof.loc[alternative][
                            cat_idx
                        ] = credibility_alt_prof.loc[alternative][category]
                    if (
                        credibility_prof_alt.loc[category][alternative]
                        > credibility_acc_prof_alt.loc[cat_idx][alternative]
                    ):
                        credibility_acc_prof_alt.loc[cat_idx][
                            alternative
                        ] = credibility_prof_alt.loc[category][alternative]
        except KeyError as exc:
            raise exceptions.SortingError(
                "Characteristic profiles list contains profile which "
                "does not exists in crisp outranking tables."
            ) from exc
        except TypeError as exc:
            if not isinstance(profile, Iterable):
                exc.args = (
                    "Value of the boundary profiles series must be an iterable, "
                    f"but got {type(profile).__name__} instead.",
                )
            raise

    crisp_table = (
        crisp_cut(credibility_acc_alt_prof, 0.7),
        crisp_cut(credibility_acc_prof_alt, 0.7),
    )
    new_profiles = pd.Series(
        characteristic_profiles.index.values, index=characteristic_profiles.index.values
    )
    return assign_tri_c_class(
        crisp_table[0],
        crisp_table[1],
        credibility_acc_alt_prof,
        credibility_acc_prof_alt,
        new_profiles,
    )


def assign_tri_rc_class(
    crisp_outranking_alt_prof: pd.DataFrame,
    crisp_outranking_prof_alt: pd.DataFrame,
    credibility_alt_prof: pd.DataFrame,
    credibility_prof_alt: pd.DataFrame,
    characteristic_profiles: pd.Series,
) -> pd.Series:
    """Implements the descending and ascending assignment rules for set of alternatives,
    based on crisp outranking relation, credibility tables and characteristic profiles.
    Unlike in `assign_tri_c_class` module, it is possible to clearly indicate the worst
    and the best class for each alternative.

    :param crisp_outranking_alt_prof: crisp outranking relation table alternatives-profiles
    :param crisp_outranking_prof_alt: crisp outranking relation table profiles-alternatives
    :param credibility_alt_prof: credibility table alternatives-profiles
    :param credibility_prof_alt: credibility table profiles-alternatives
    :param characteristic_profiles: profiles which characterize classes in ascending order

    :return: `pandas.Series` of pairs with the worst and the best class
        indicated for each alternative
    """
    assignment = pd.Series([], dtype=pd.StringDtype(storage=None))

    _check_tables_ap(
        crisp_outranking_alt_prof,
        crisp_outranking_prof_alt,
        index_validation_function=_check_index_value_binary,
        name="crisp outranking",
    )
    _check_tables_ap(
        credibility_alt_prof,
        credibility_prof_alt,
        index_validation_function=_check_index_value_interval,
        name="credibility",
    )
    try:
        _unique_names(characteristic_profiles.keys(), names_type="categories")
    except AttributeError as exc:
        raise TypeError(
            f"Wrong characteristic_profiles type. Expected {pd.Series.__name__}, "
            f"but got {type(characteristic_profiles).__name__} instead."
        ) from exc

    assignments_descending = []
    assignments_ascending = []
    for alternative in crisp_outranking_alt_prof.index.values:
        found_descending = False
        try:
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
        except KeyError as exc:
            raise exceptions.SortingError(
                "Characteristic profiles list contains profile which "
                "does not exists in crisp outranking and credibility tables."
            ) from exc

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
