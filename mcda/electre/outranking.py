"""This module implements methods to make an outranking."""
from enum import Enum
from typing import Optional, Tuple, Union

import pandas as pd

from ..core.aliases import NumericValue
from ._validate import _check_index_value_interval


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
                for alt_name_b in credibility_table.index
            ]
            for alt_name_a in credibility_table.index
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
    _check_index_value_interval(
        concordance_cutting_level, "cutting level", minimal_val=0.5
    )

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
                for alt_name_b in concordance_comprehensive_table.index
            ]
            for alt_name_a in concordance_comprehensive_table.index
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
                for alt_name_b in concordance_comprehensive_table.index
            ]
            for alt_name_a in concordance_comprehensive_table.index
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
                    for profile_name in crisp_outranking_table_profiles.index
                ]
                for alt_name in crisp_outranking_table.index
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
                    for alt_name in crisp_outranking_table.index
                ]
                for profile_name in crisp_outranking_table_profiles.index
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
                for alt_name_b in crisp_outranking_table.index
            ]
            for alt_name_a in crisp_outranking_table.index
        ],
        index=crisp_outranking_table.index,
        columns=crisp_outranking_table.index,
    )
