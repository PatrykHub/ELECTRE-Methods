from typing import List

import numpy as np
import pandas as pd
import pytest

from mcda.electre.outranking import (
    _get_maximal_credibility_index,
    _get_minimal_credibility_index,
    alternative_qualities,
    crisp_outranking_relation_distillation,
    distillation,
    final_ranking_matrix,
    median_order,
    ranks,
)
from mcda.electre.utils import order_to_outranking_matrix


@pytest.fixture
def credibility_matrix() -> pd.DataFrame:
    alt_names: List[str] = ["P1", "P2", "P3", "P4", "P5"]
    credibility_matrix = pd.DataFrame(
        [
            [1.0, 0.0, 1.0, 0.8, 1.0],
            [0.0, 1.0, 0.0, 0.9, 0.67],
            [0.6, 0.0, 1.0, 0.6, 0.8],
            [0.25, 0.8, 0.67, 1.0, 0.85],
            [0.67, 0.0, 0.8, 0.8, 1.0],
        ],
        index=alt_names,
        columns=alt_names,
    )
    np.fill_diagonal(credibility_matrix.values, 0)
    return credibility_matrix


def test_get_maximal_credibility_index(credibility_matrix) -> None:
    assert _get_maximal_credibility_index(credibility_matrix) == 1.0


def test_get_minimal_credibility_index(credibility_matrix) -> None:
    assert _get_minimal_credibility_index(credibility_matrix, 1.0) == 0.8


@pytest.mark.parametrize(
    (
        "credibility_pair_value",
        "credibility_pair_reverse_value",
        "minimal_credibility_index",
        "expected",
    ),
    ((0.7, 0.3, 0.6, 1), (0.85, 0.6, 0.6, 1), (0.7, 0.58, 0.6, 0)),
)
def test_crisp_outranking_relation_distillation(
    credibility_pair_value,
    credibility_pair_reverse_value,
    minimal_credibility_index,
    expected,
) -> None:
    assert (
        crisp_outranking_relation_distillation(
            credibility_pair_value,
            credibility_pair_reverse_value,
            minimal_credibility_index,
        )
        == expected
    )


def test_alternative_qualities(credibility_matrix) -> None:
    alt_names: List[str] = ["P1", "P2", "P3", "P4", "P5"]
    result, _ = alternative_qualities(credibility_matrix)

    assert result.equals(pd.Series([2, 0, -1, 0, -1], index=alt_names))


def test_distillation(credibility_matrix) -> None:
    expected_downward = pd.Series([["P1"], ["P2"], ["P3", "P4", "P5"]])
    expected_upward = pd.Series([["P3", "P5"], ["P4"], ["P1", "P2"]])

    downward = distillation(credibility_matrix)

    for i in range(len(downward)):
        assert downward[i + 1] == expected_downward[i]

    upward = distillation(credibility_matrix, upward_order=True)

    for i in range(len(upward)):
        assert upward[i + 1] == expected_upward[i]


@pytest.fixture
def downward_order_matrix() -> pd.DataFrame:
    alt_names_descending: List[str] = ["GER", "ITA", "BEL", "AUT", "FRA"]
    return pd.DataFrame(
        [
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ],
        index=alt_names_descending,
        columns=alt_names_descending,
    )


@pytest.fixture
def upward_order_matrix() -> pd.DataFrame:
    alt_names_ascending: List[str] = ["FRA", "GER", "ITA", "BEL", "AUT"]
    return pd.DataFrame(
        [
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
        ],
        index=alt_names_ascending,
        columns=alt_names_ascending,
    )


@pytest.fixture
def downward_order() -> pd.Series:
    return pd.Series([["GER"], ["ITA"], ["BEL", "AUT", "FRA"]])


@pytest.fixture
def upward_order() -> pd.Series:
    return pd.Series([["FRA"], ["GER"], ["ITA"], ["BEL"], ["AUT"]])


def test_order_to_outranking_matrix_downward(downward_order, downward_order_matrix) -> None:
    assert order_to_outranking_matrix(downward_order).equals(downward_order_matrix)


def test_order_to_outranking_matrix_upward(upward_order, upward_order_matrix) -> None:
    assert order_to_outranking_matrix(upward_order).equals(upward_order_matrix)


@pytest.fixture
def final_ranking() -> pd.DataFrame:
    alt_names: List[str] = ["AUT", "BEL", "FRA", "GER", "ITA"]
    return pd.DataFrame(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 0, 1],
        ],
        index=alt_names,
        columns=alt_names,
    )


def test_final_ranking_matrix(downward_order, upward_order, final_ranking) -> None:
    assert final_ranking_matrix(downward_order, upward_order).equals(final_ranking)


@pytest.fixture
def ranks_ranking() -> pd.Series:
    return pd.Series([["FRA", "GER"], ["ITA"], ["BEL"], ["AUT"]], index=[1, 2, 3, 4])


def test_ranks(final_ranking, ranks_ranking) -> None:
    assert ranks(final_ranking).equals(ranks_ranking)


def test_median_order(ranks_ranking, downward_order, upward_order) -> None:
    assert median_order(ranks_ranking, downward_order, upward_order).equals(
        pd.Series([["GER"], ["FRA"], ["ITA"], ["BEL"], ["AUT"]], index=[1, 2, 3, 4, 5])
    )
