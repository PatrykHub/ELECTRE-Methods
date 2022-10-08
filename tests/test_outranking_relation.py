from typing import List, Tuple, Union

import pandas as pd
import pytest
from mcda.electre.outranking import (
    OutrankingRelation,
    crisp_outranking_coal,
    crisp_outranking_coal_marginal,
    crisp_outranking_cut,
    crisp_outranking_cut_marginal,
    crisp_outranking_Is,
    crisp_outranking_Is_marginal,
    outranking_relation,
    outranking_relation_marginal,
)


@pytest.mark.parametrize(
    ("credibility", "cutting_level", "expected"),
    (
        (0, 0.5, False),
        (1, 0.5, True),
        (0.6786, 0.6786, True),
    ),
)
def test_crisp_outranking_cut_marginal(
    credibility,
    cutting_level,
    expected: Union[bool, type],
) -> None:
    assert crisp_outranking_cut_marginal(credibility, cutting_level) == expected


def test_crisp_outranking_cut() -> None:
    alt_names: List[str] = ["A1", "A2", "A3"]
    assert crisp_outranking_cut(
        pd.DataFrame(
            [
                [1, 0.996, 0.673],
                [0.208, 1, 0.410],
                [0.530, 0.648, 1],
            ],
            index=alt_names,
            columns=alt_names,
        ),
        0.673,
    ).equals(
        pd.DataFrame(
            [[True, True, True], [False, True, False], [False, False, True]],
            index=alt_names,
            columns=alt_names,
        )
    )


@pytest.mark.parametrize(
    ("concordance", "discordance", "cutting_level", "expected"),
    (
        (0.73, 1, 0.73, False),
        (0.73, 0, 0.73, True),
        (0.89, 0, 0.73, True),
    ),
)
def test_crisp_outranking_Is_marginal(
    concordance, discordance, cutting_level, expected: Union[bool, type]
) -> None:
    assert (
        crisp_outranking_Is_marginal(concordance, discordance, cutting_level)
        == expected
    )


def test_crisp_outranking_Is() -> None:
    alt_names: List[str] = ["A1", "A2"]
    concordance_comprehensive = pd.DataFrame(
        [[1.0, 0.74], [0.74, 1.0]], index=alt_names, columns=alt_names
    )
    discordance_comprehensive_bin = pd.DataFrame(
        [[0, 1], [0, 0]], index=alt_names, columns=alt_names
    )

    assert crisp_outranking_Is(
        concordance_comprehensive,
        discordance_comprehensive_bin,
        0.74,
    ).equals(
        pd.DataFrame([[True, False], [True, True]], index=alt_names, columns=alt_names)
    )


@pytest.mark.parametrize(
    (
        "concordance",
        "discordance",
        "conc_cutting_level",
        "disc_cutting_level",
        "expected",
    ),
    (
        (1, 0, 0.5, 0.000001, True),
        (0.86, 0.025, 0.85, 0.1, True),
        (0.86, 0.125, 0.85, 0.1, False),
    ),
)
def test_crisp_outranking_coal_marginal(
    concordance,
    discordance,
    conc_cutting_level,
    disc_cutting_level,
    expected: Union[type, bool],
) -> None:
    assert (
        crisp_outranking_coal_marginal(
            concordance, discordance, conc_cutting_level, disc_cutting_level
        )
        == expected
    )


def test_crisp_outranking_coal() -> None:
    alt_names: List[str] = ["A1", "A2", "A3"]
    concordance_comprehensive = pd.DataFrame(
        [[1.0, 0.788, 0.504], [0.07, 1.0, 0.038], [0.925, 0.404, 1.0]],
        index=alt_names,
        columns=alt_names,
    )
    discordance_comprehensive = pd.DataFrame(
        [[0.0, 0.536, 0.125], [0.632, 0.0, 0.499], [0.370, 0.259, 0.0]],
        index=alt_names,
        columns=alt_names,
    )
    assert crisp_outranking_coal(
        concordance_comprehensive,
        discordance_comprehensive,
        0.504,
        0.370,
    ).equals(
        pd.DataFrame(
            [[True, False, True], [False, True, False], [False, False, True]],
            index=alt_names,
            columns=alt_names,
        )
    )


@pytest.mark.parametrize(
    ("crisp_outranking_1", "crisp_outranking_2", "expected"),
    (
        (True, False, OutrankingRelation.PQ),
        (True, True, OutrankingRelation.INDIFF),
        (False, False, OutrankingRelation.R),
        ("", "", OutrankingRelation.R),
        (False, True, None),
    ),
)
def test_outranking_relation_marginal(
    crisp_outranking_1,
    crisp_outranking_2,
    expected: Union[None, OutrankingRelation, type],
) -> None:
    if isinstance(expected, OutrankingRelation) or expected is None:
        assert (
            outranking_relation_marginal(crisp_outranking_1, crisp_outranking_2)
            == expected
        )
    else:
        with pytest.raises(expected):
            outranking_relation_marginal(crisp_outranking_1, crisp_outranking_2)


@pytest.mark.parametrize(
    ("crisp_outranking_table", "crisp_outranking_table_profiles", "expected"),
    (
        (
            pd.DataFrame(
                [[True, False, True], [False, True, False], [False, True, True]],
                index=["A1", "A2", "A3"],
                columns=["A1", "A2", "A3"],
            ),
            None,
            pd.DataFrame(
                [
                    [
                        OutrankingRelation.INDIFF,
                        OutrankingRelation.R,
                        OutrankingRelation.PQ,
                    ],
                    [OutrankingRelation.R, OutrankingRelation.INDIFF, None],
                    [None, OutrankingRelation.PQ, OutrankingRelation.INDIFF],
                ],
                index=["A1", "A2", "A3"],
                columns=["A1", "A2", "A3"],
            ),
        ),
        (
            pd.DataFrame(
                [[False, True, True], [False, True, False]],
                index=["A1", "A2"],
                columns=["P1", "P2", "P3"],
            ),
            pd.DataFrame(
                [[True, False], [False, False], [False, True]],
                index=["P1", "P2", "P3"],
                columns=["A1", "A2"],
            ),
            (
                pd.DataFrame(
                    [
                        [None, OutrankingRelation.PQ, OutrankingRelation.PQ],
                        [OutrankingRelation.R, OutrankingRelation.PQ, None],
                    ],
                    index=["A1", "A2"],
                    columns=["P1", "P2", "P3"],
                ),
                pd.DataFrame(
                    [
                        [OutrankingRelation.PQ, OutrankingRelation.R],
                        [None, None],
                        [None, OutrankingRelation.PQ],
                    ],
                    index=["P1", "P2", "P3"],
                    columns=["A1", "A2"],
                ),
            ),
        ),
    ),
)
def test_outranking_relation(
    crisp_outranking_table,
    crisp_outranking_table_profiles,
    expected: Union[Tuple, pd.DataFrame],
) -> None:
    result = outranking_relation(
        crisp_outranking_table, crisp_outranking_table_profiles
    )

    if isinstance(expected, tuple):
        assert len(result) == 2
        assert result[0].equals(expected[0])
        assert result[1].equals(expected[1])
    else:
        assert isinstance(result, pd.DataFrame)
        assert result.equals(expected)
