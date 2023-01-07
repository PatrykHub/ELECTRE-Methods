from typing import List

import numpy as np
import pandas as pd
import pytest

from mcda.electre.outranking.ranking import (
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


@pytest.fixture
def credibility_matrix_advanced() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1.0, 1.0, 0.9977, 0.606, 1.0, 0.9977, 0.8175, 1.0, 0.803, 0.8775, 0.548, 0.5318],
            [1.0, 1.0, 0.9977, 0.4047, 0.9973, 0.9816, 0.7762, 1.0, 0.6612, 0.8775, 0.4951, 0.443],
            [
                0.7734,
                0.8946,
                1.0,
                0.2775,
                0.8041,
                0.8014,
                0.6083,
                0.916,
                0.499,
                0.8775,
                0.382,
                0.3598,
            ],
            [0.85, 0.85, 0.9317, 1.0, 0.85, 0.9317, 0.7739, 0.916, 0.8742, 0.8775, 0.7464, 0.7912],
            [
                0.9786,
                0.9854,
                0.9977,
                0.5903,
                1.0,
                0.9816,
                0.7184,
                1.0,
                0.7288,
                0.8775,
                0.4546,
                0.3839,
            ],
            [0.8946, 0.9014, 1.0, 0.5999, 0.916, 1.0, 0.7488, 0.916, 0.7128, 0.8775, 0.464, 0.464],
            [1.0, 1.0, 1.0, 0.691, 1.0, 1.0, 1.0, 1.0, 0.803, 0.8775, 0.6875, 0.6875],
            [
                0.5299,
                0.7279,
                0.7849,
                0.2795,
                0.7057,
                0.5066,
                0.3148,
                1.0,
                0.3638,
                0.8118,
                0.2394,
                0.1979,
            ],
            [0.8692, 0.934, 1.0, 0.773, 0.9352, 0.9352, 0.7297, 1.0, 1.0, 0.8775, 0.7647, 0.8393],
            [
                0.5803,
                0.8033,
                0.9186,
                0.3486,
                0.7385,
                0.5779,
                0.4384,
                0.916,
                0.5775,
                1.0,
                0.4959,
                0.3625,
            ],
            [0.8692, 0.934, 0.9317, 0.773, 0.8692, 0.8669, 0.8669, 1.0, 0.9688, 1.0, 1.0, 0.9977],
            [0.8692, 0.934, 0.9537, 0.7835, 0.8822, 0.8822, 0.8692, 1.0, 0.993, 1.0, 1.0, 1.0],
        ]
    )


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


def test_distillation_upward(credibility_matrix) -> None:
    expected_upward = pd.Series([["P3", "P5"], ["P4"], ["P1", "P2"]])
    upward = distillation(credibility_matrix, upward_order=True)

    for i in range(len(upward)):
        assert upward[i + 1] == expected_upward[i]


def test_distillation_downward(credibility_matrix) -> None:
    expected_downward = pd.Series([["P1"], ["P2"], ["P3", "P4", "P5"]])
    downward = distillation(credibility_matrix)

    for i in range(len(downward)):
        assert downward[i + 1] == expected_downward[i]


def test_distillation_upward_advanced(credibility_matrix_advanced) -> None:
    credibility_matrix_advanced.index = [
        f"V{x}" for x in range(1, len(credibility_matrix_advanced) + 1)
    ]
    credibility_matrix_advanced.columns = [
        f"V{x}" for x in range(1, len(credibility_matrix_advanced) + 1)
    ]

    expected = pd.Series(
        [
            ["V11", "V12", "V4"],
            ["V7", "V9"],
            ["V1"],
            ["V2", "V5", "V6"],
            ["V10"],
            ["V3"],
            ["V8"],
        ],
        index=[7, 6, 5, 4, 3, 2, 1],
    )
    results = distillation(credibility_matrix_advanced, upward_order=True)
    for expected_list, result_list in zip(expected, results):
        assert set(expected_list) == set(result_list)


def test_distillation_downward_advanced(credibility_matrix_advanced) -> None:
    credibility_matrix_advanced.index = [
        f"V{x}" for x in range(1, len(credibility_matrix_advanced) + 1)
    ]
    credibility_matrix_advanced.columns = [
        f"V{x}" for x in range(1, len(credibility_matrix_advanced) + 1)
    ]

    expected = pd.Series(
        [
            ["V11", "V12"],
            ["V4", "V7"],
            ["V9"],
            ["V1", "V6"],
            ["V5"],
            ["V2"],
            ["V10", "V3", "V8"],
        ],
        index=[1, 2, 3, 4, 5, 6, 7],
    )
    results = distillation(credibility_matrix_advanced)
    for expected_list, result_list in zip(expected, results):
        assert set(expected_list) == set(result_list)


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
    return pd.Series([["GER"], ["ITA"], ["BEL", "AUT", "FRA"]], index=[1, 2, 3])


@pytest.fixture
def upward_order() -> pd.Series:
    return pd.Series([["FRA"], ["GER"], ["ITA"], ["BEL"], ["AUT"]], index=[1, 2, 3, 4, 5])


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
