"""This module implements methods to make an outranking."""
from enum import Enum
from typing import List, Optional, Tuple, Union

from ..core.aliases import NumericValue
from ._validate import _check_indice_value_interval


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
    _check_indice_value_interval(credibility, "credibility")
    _check_indice_value_interval(cutting_level, "cutting level", minimal_val=0.5)
    return credibility >= cutting_level


def crisp_outranking_cut(
    credibiliy_table: List[List[NumericValue]],
    cutting_level: NumericValue,
) -> List[List[bool]]:
    """Constructs crisp outranking relations,
    based on credibility values.

    :param credibiliy_table: table with credibility values
    :param cutting_level: value from [0.5, 1] interval

    :return: Boolean table the same size as the credibility table
    """
    return [
        [
            crisp_outranking_cut_marginal(credibility, cutting_level)
            for credibility in credibility_row
        ]
        for credibility_row in credibiliy_table
    ]


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
    _check_indice_value_interval(concordance_comprehensive, "comprehensive concordance")
    _check_indice_value_interval(
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
    concordance_comprehensive_table: List[List[NumericValue]],
    discordance_comprehensive_bin_table: List[List[int]],
    concordance_cutting_level: NumericValue,
) -> List[List[bool]]:
    """Constructs a crisp outranking relations, based on
    comprehensive concordance C(a, b) and comprehensive
    binary discordance D(a, b) indices.

    :param concordance_comprehensive_table: table with comprehensive concordance indices
    :param discordance_comprehensive_bin_table: table with comprehensive binary
    discordance indices
    :param concordance_cutting_level: concordance majority threshold (cutting level)

    :return: Boolean table the same size as the concordance and discordance tables
    """
    return [
        [
            crisp_outranking_Is_marginal(
                concordance_comprehensive,
                discordance_comprehensive_bin,
                concordance_cutting_level,
            )
            for concordance_comprehensive, discordance_comprehensive_bin in zip(
                concordance_tabe_row, discordance_table_row
            )
        ]
        for concordance_tabe_row, discordance_table_row in zip(
            concordance_comprehensive_table, discordance_comprehensive_bin_table
        )
    ]


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
    _check_indice_value_interval(concordance_comprehensive, "comprehensive concordance")
    _check_indice_value_interval(discordance_comprehensive, "comprehensive discordance")
    _check_indice_value_interval(
        concordance_cutting_level, "concordance majority threshold", minimal_val=0.5
    )
    _check_indice_value_interval(
        discordance_cutting_level, "discordance majority threshold", include_min=False
    )
    return (
        concordance_comprehensive >= concordance_cutting_level
        and discordance_comprehensive < discordance_cutting_level
    )


def crisp_outranking_coal(
    concordance_comprehensive_table: List[List[NumericValue]],
    discordance_comprehensive_table: List[List[NumericValue]],
    concordance_cutting_level: NumericValue,
    discordance_cutting_level: NumericValue,
) -> List[List[bool]]:
    """Constructs a crisp outranking relations, based on
    comprehensive concordance C(a, b) and comprehensive
    discordance D(a, b) indices.

    :param concordance_comprehensive_table: comprehensive concordance table
    :param discordance_comprehensive_table: comprehensive discordance table
    :param concordance_cutting_level: concordance majority threshold (cutting level)
    :param discordance_cutting_level: discordance majority threshold (cutting level)

    :return: Boolean table the same size as the concordance and discordance tables
    """
    return [
        [
            crisp_outranking_coal_marginal(
                concordance_comprehensive,
                discordance_comprehensive,
                concordance_cutting_level,
                discordance_cutting_level,
            )
            for concordance_comprehensive, discordance_comprehensive in zip(
                concordance_table_row, discordance_table_row
            )
        ]
        for concordance_table_row, discordance_table_row in zip(
            concordance_comprehensive_table, discordance_comprehensive_table
        )
    ]


def outranking_relation_marginal(
    crisp_outranking_1: bool,
    crisp_outranking_2: bool,
) -> Optional[OutrankingRelation]:
    """Aggregates the crisp outranking relations

    :param crisp_outranking_1: crisp outranking relation of (a, b) alternatives
    :param crisp_outranking_2: crisp outranking relation of (b, a) alternatives

    :return:
        * None, if b is preferred to a
        * OutrankingRelation enum
    """
    if crisp_outranking_1 and crisp_outranking_2:
        return OutrankingRelation.INDIFF

    if crisp_outranking_1 and not crisp_outranking_2:
        return OutrankingRelation.PQ

    if not crisp_outranking_1 and not crisp_outranking_2:
        return OutrankingRelation.R

    return None


def outranking_relation(
    crisp_outranking_table: List[List[bool]],
    crisp_outranking_table_profiles: Optional[List[List[bool]]],
) -> Union[
    List[List[Optional[OutrankingRelation]]],
    Tuple[List[List[Optional[OutrankingRelation]]], ...],
]:
    """Aggregates crisp outranking relations.

    :param crisp_outranking_table: table with crisp relations
    between alternatives or between alternatives and profiles.
    :param crisp_outranking_table_profiles: table with crisp relations
    between profiles and alternatives

    :return: one table with outranking relations between alternatives, if only
    one table was provided, in other case the function will return two
    tables - first one with alternatives - profiles, second one with
    profiles - alternatives comparision.
    """
    if crisp_outranking_table_profiles is not None:
        return [
            [
                outranking_relation_marginal(
                    crisp_outranking_table[i][j], crisp_outranking_table_profiles[j][i]
                )
                for j in range(len(crisp_outranking_table[i]))
            ]
            for i in range(len(crisp_outranking_table))
        ], [
            [
                outranking_relation_marginal(
                    crisp_outranking_table_profiles[i][j], crisp_outranking_table[j][i]
                )
                for j in range(len(crisp_outranking_table_profiles[i]))
            ]
            for i in range(len(crisp_outranking_table_profiles))
        ]
    return [
        [
            outranking_relation_marginal(
                crisp_outranking_table[i][j], crisp_outranking_table[j][i]
            )
            for j in range(len(crisp_outranking_table[i]))
        ]
        for i in range(len(crisp_outranking_table))
    ]
