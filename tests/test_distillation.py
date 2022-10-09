from typing import List

import numpy as np
import pandas as pd
import pytest

from mcda.electre.outranking import (alternative_qualities,
                                     crisp_outranking_relation_distillation,
                                     distillation,
                                     get_maximal_credibility_index,
                                     get_minimal_credibility_index)


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
    assert get_maximal_credibility_index(credibility_matrix) == 1.0


def test_get_minimal_credibility_index(credibility_matrix) -> None:
    assert get_minimal_credibility_index(credibility_matrix, 1.0) == 0.8


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
        assert downward[i] == expected_downward[i]

    upward = distillation(credibility_matrix, upward_order=True)

    for i in range(len(upward)):
        assert upward[i] == expected_upward[i]