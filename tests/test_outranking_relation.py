from typing import Collection, Union

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
    assert crisp_outranking_cut(
        [
            [1, 0.996, 0.673],
            [0.208, 1, 0.410],
            [0.530, 0.648, 1],
        ],
        0.673,
    ) == [[True, True, True], [False, True, False], [False, False, True]]


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
    assert crisp_outranking_Is(
        [[1.0, 0.74], [0.74, 1.0]],
        [[0, 1], [0, 0]],
        0.74,
    ) == [[True, False], [True, True]]


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
    assert crisp_outranking_coal(
        [[1.0, 0.788, 0.504], [0.07, 1.0, 0.038], [0.925, 0.404, 1.0]],
        [[0.0, 0.536, 0.125], [0.632, 0.0, 0.499], [0.370, 0.259, 0.0]],
        0.504,
        0.370,
    ) == [[True, False, True], [False, True, False], [False, False, True]]


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
            [[True, False, True], [False, True, False], [False, True, True]],
            None,
            [
                [
                    OutrankingRelation.INDIFF,
                    OutrankingRelation.R,
                    OutrankingRelation.PQ,
                ],
                [OutrankingRelation.R, OutrankingRelation.INDIFF, None],
                [None, OutrankingRelation.PQ, OutrankingRelation.INDIFF],
            ],
        ),
        (
            [[False, True, True], [False, True, False]],
            [[True, False], [False, False], [False, True]],
            (
                [
                    [None, OutrankingRelation.PQ, OutrankingRelation.PQ],
                    [OutrankingRelation.R, OutrankingRelation.PQ, None],
                ],
                [
                    [OutrankingRelation.PQ, OutrankingRelation.R],
                    [None, None],
                    [None, OutrankingRelation.PQ],
                ],
            ),
        ),
    ),
)
def test_outranking_relation(
    crisp_outranking_table,
    crisp_outranking_table_profiles,
    expected: Union[type, Collection],
) -> None:
    if isinstance(expected, Collection):
        result = outranking_relation(
            crisp_outranking_table, crisp_outranking_table_profiles
        )
        assert result == expected
        if crisp_outranking_table_profiles is not None:
            assert len(result) == 2
    else:
        with pytest.raises(expected):
            outranking_relation(crisp_outranking_table, crisp_outranking_table_profiles)
