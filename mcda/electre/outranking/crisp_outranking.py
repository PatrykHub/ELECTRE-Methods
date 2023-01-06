"""This module implements methods to transform valued outranking relation into crisp one."""
from enum import Enum
from typing import Optional, Tuple, Union

import pandas as pd

from mcda.core.aliases import NumericValue

from .. import exceptions
from .._validation import (
    _check_index_value_binary,
    _check_index_value_interval,
    _consistent_df_indexing,
)


class OutrankingRelation(Enum):
    """Definition of preference relations (also possible incomparability).

    :param Enum: preference relation shortcut
    """

    INDIFF = "INDIFFERENCE"
    PQ = "STRONG OR WEAK PREFERENCE"
    R = "INCOMPARABILITY"


def crisp_outranking_cut_marginal(
    credibility: NumericValue,
    cutting_level: NumericValue,
) -> bool:
    """Computes the single crisp outranking relation value :math:`S^{CUT}(a, b)`,
    based on credibility value :math:`S(a, b)` and cutting level :math:`\\lambda`.

    :param credibility: credibility value :math:`S(a, b)`, value from [0, 1] interval
    :param cutting_level: majority threshold, value from [0.5, 1] interval

    :return: crisp outranking relation value, ``True`` if a outranks b, ``False`` otherwise
    """
    _check_index_value_interval(credibility, "credibility")
    _check_index_value_interval(cutting_level, "cutting level", minimal_val=0.5)
    return credibility >= cutting_level


def crisp_outranking_cut(
    credibility_table: pd.DataFrame,
    cutting_level: NumericValue,
) -> pd.DataFrame:
    """Constructs a crisp outranking relation :math:`S^{CUT}`, based on credibility table `S`
    and cutting level :math:`\\lambda`.

    :param credibility_table: credibility table :math:`S` with values from [0, 1] interval
    :param cutting_level: majority threshold, value from [0.5, 1] interval

    :return: crisp outranking relation table, ``True`` if a outranks b, ``False`` otherwise
    """
    _consistent_df_indexing(credibility_table=credibility_table)
    return pd.DataFrame(
        [
            [
                crisp_outranking_cut_marginal(
                    credibility_table.loc[alt_name_a][alt_name_b], cutting_level
                )
                for alt_name_b in credibility_table.columns.values
            ]
            for alt_name_a in credibility_table.index.values
        ],
        index=credibility_table.index,
        columns=credibility_table.columns,
    )


def crisp_outranking_Is_marginal(
    concordance_comprehensive: NumericValue,
    discordance_comprehensive_bin: int,
    concordance_cutting_level: NumericValue,
) -> bool:
    """Computes the single crisp outranking relation :math:`S^{Is}(a, b)`, based on
    comprehensive concordance index :math:`C(a, b)`,comprehensive  binary discordance
    index :math:`D^V(a, b)` and concordance cutting level :math:`\\lambda^C`.

    :param concordance_comprehensive: comprehensive concordance index :math:`C(a, b)`,
        value from [0, 1] interval
    :param discordance_comprehensive_bin: comprehensive binary discordance index :math:`D^V(a, b)`
    :param concordance_cutting_level: concordance majority threshold (cutting level),
        value from [0.5, 1] interval

    :return: crisp outranking relation value, ``True`` if a outranks b, ``False`` otherwise
    """
    _check_index_value_interval(concordance_comprehensive, "comprehensive concordance")
    _check_index_value_interval(
        concordance_cutting_level, "cutting level", minimal_val=0.5
    )
    _check_index_value_binary(discordance_comprehensive_bin, "binary discordance")
    return (
        concordance_comprehensive >= concordance_cutting_level
        and discordance_comprehensive_bin == 0
    )


def crisp_outranking_Is(
    concordance_comprehensive: pd.DataFrame,
    discordance_comprehensive_bin: pd.DataFrame,
    concordance_cutting_level: NumericValue,
) -> pd.DataFrame:
    """Constructs a crisp outranking relation :math:`S^{Is}`, based on comprehensive
    concordance :math:`C`, comprehensive binary discordance :math:`D^V`
    and concordance cutting level :math:`\\lambda^C`.

    :param concordance_comprehensive: comprehensive concordance :math:`C`,
        with values from [0, 1] interval
    :param discordance_comprehensive_bin: comprehensive binary discordance :math:`D^V`
    :param concordance_cutting_level: concordance majority threshold (cutting level),
        value from [0.5, 1] interval

    :return: crisp outranking relation table, ``True`` if a outranks b, ``False`` otherwise
    """
    _consistent_df_indexing(
        concordance_comprehensive=concordance_comprehensive,
        discordance_comprehensive_bin=discordance_comprehensive_bin,
    )
    return pd.DataFrame(
        [
            [
                crisp_outranking_Is_marginal(
                    concordance_comprehensive.loc[alt_name_a][alt_name_b],
                    discordance_comprehensive_bin.loc[alt_name_a][alt_name_b],
                    concordance_cutting_level,
                )
                for alt_name_b in concordance_comprehensive.columns.values
            ]
            for alt_name_a in concordance_comprehensive.index.values
        ],
        index=concordance_comprehensive.index,
        columns=concordance_comprehensive.columns,
    )


def crisp_outranking_coal_marginal(
    concordance_comprehensive: NumericValue,
    discordance_comprehensive: NumericValue,
    concordance_cutting_level: NumericValue,
    discordance_cutting_level: NumericValue,
) -> bool:
    """Computes the single crisp outranking relation :math:`S^{COAL}(a, b)`, based on
    comprehensive concordance :math:`C(a, b)`and comprehensive discordance :math:`D(a, b)`
    indices considering concordance :math:`\\lambda^C` and discordance cutting levels
    :math:`\\lambda^D`.

    :param concordance_comprehensive: comprehensive concordance index :math:`C(a, b)`,
        value from [0, 1] interval
    :param discordance_comprehensive: comprehensive discordance index :math:`D(a, b)`,
        value from [0, 1] interval
    :param concordance_cutting_level: concordance majority threshold (cutting level),
        value from [0.5, 1] interval
    :param discordance_cutting_level: discordance majority threshold (cutting level),
        value from [0, 1] interval

    :return: crisp outranking relation value, ``True`` if a outranks b, ``False`` otherwise
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
    concordance_comprehensive: pd.DataFrame,
    discordance_comprehensive: pd.DataFrame,
    concordance_cutting_level: NumericValue,
    discordance_cutting_level: NumericValue,
) -> pd.DataFrame:
    """Constructs a crisp outranking relation :math:`S^{COAL}`, based on comprehensive
    concordance :math:`C`and comprehensive discordance :math:`D` indices considering
    concordance :math:`\\lambda^C` and discordance cutting levels :math:`\\lambda^D`.

    :param concordance_comprehensive: comprehensive concordance :math:`C`,
        with values from [0, 1] interval
    :param discordance_comprehensive: comprehensive discordance :math:`D`,
        with values from [0, 1] interval
    :param concordance_cutting_level: concordance majority threshold (cutting level),
        value from [0.5, 1] interval
    :param discordance_cutting_level: discordance majority threshold (cutting level),
        value from [0.5, 1] interval

    :return: crisp outranking relation table, ``True`` if a outranks b, ``False`` otherwise
    """
    _consistent_df_indexing(
        concordance_comprehensive=concordance_comprehensive,
        discordance_comprehensive=discordance_comprehensive,
    )
    return pd.DataFrame(
        [
            [
                crisp_outranking_coal_marginal(
                    concordance_comprehensive.loc[alt_name_a][alt_name_b],
                    discordance_comprehensive.loc[alt_name_a][alt_name_b],
                    concordance_cutting_level,
                    discordance_cutting_level,
                )
                for alt_name_b in concordance_comprehensive.columns.values
            ]
            for alt_name_a in concordance_comprehensive.index.values
        ],
        index=concordance_comprehensive.index,
        columns=concordance_comprehensive.columns,
    )


def outranking_relation_marginal(
    crisp_outranking_ab: bool,
    crisp_outranking_ba: bool,
) -> Optional[OutrankingRelation]:
    """Constructs a crisp outranking relation for one pair alternative-alternative
    or alternative-profile.

    :param crisp_outranking_ab: crisp outranking value of alternative
    :param crisp_outranking_ba: crisp outranking value of alternative/profile

    :return:
        * ``None``, if b is preferred to a
        * `OutrankingRelation` Enum, otherwise
    """
    _check_index_value_binary(crisp_outranking_ab, name="crisp relation")
    _check_index_value_binary(crisp_outranking_ba, name="crisp relation")
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
    """Constructs a crisp outranking relation between alternatives or alternatives-profiles.

    :param crisp_outranking_table: crisp outranking table of alternatives
    :param crisp_outranking_table_profiles: optional crisp outranking table of profiles,
        defaults to ``None``

    :raises exceptions.InconsistentDataFrameIndexingError: _description_
    :raises exceptions.InconsistentDataFrameIndexingError: _description_
    .. todo::
        describe exception

    :return:
        * if `crisp_outranking_table_profiles` argument is set to ``None``, the function
          will return a single `pandas.DataFrame` object with outranking relation
          for all alternatives pairs
        * otherwise, the function will return a ``tuple`` object with two
          `pandas.DataFrame` objects inside, where the first one contains
          outranking relation alternatives-profiles, and the second one
          outranking relation profiles-alternatives, respectively
    """
    _consistent_df_indexing(crisp_outranking_table=crisp_outranking_table)
    if crisp_outranking_table_profiles is not None:
        _consistent_df_indexing(
            crisp_outranking_table_profiles=crisp_outranking_table_profiles
        )

        if set(crisp_outranking_table_profiles.index.values) != set(
            crisp_outranking_table.columns.values
        ):
            raise exceptions.InconsistentDataFrameIndexingError(
                "Profiles names are not the same for provided tables."
            )

        if set(crisp_outranking_table_profiles.columns.values) != set(
            crisp_outranking_table.index.values
        ):
            raise exceptions.InconsistentDataFrameIndexingError(
                "Alternatives names are not the same for provided tables."
            )

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
