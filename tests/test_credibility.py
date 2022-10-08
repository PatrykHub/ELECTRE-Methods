from typing import List

import pytest

from mcda.core.aliases import NumericValue
from mcda.core.scales import PreferenceDirection, QuantitativeScale
from mcda.electre.credibility import (credibility_electre_iv,
                                      get_criteria_counts)


@pytest.fixture
def scales() -> List[QuantitativeScale]:
    return [
        QuantitativeScale(0, 100),
        QuantitativeScale(0, 10),
        QuantitativeScale(0, 1000, PreferenceDirection.MIN),
    ]


@pytest.fixture
def indifference_thresholds() -> List[NumericValue]:
    return [4, 1, 100]


@pytest.fixture
def preference_thresholds() -> List[NumericValue]:
    return [12, 2, 200]


@pytest.mark.parametrize(
    (
        "a_values",
        "b_values",
        "expected",
    ),
    (
        ([90, 4, 600], [58, 0, 200], [2, 0, 0, 0]),
        ([58, 0, 200], [90, 4, 600], [1, 0, 0, 0]),
        ([74, 8, 800], [66, 7, 400], [0, 1, 1, 0]),
    ),
)
def test_get_criteria_counts(
    a_values, b_values, scales, indifference_thresholds, preference_thresholds, expected
) -> None:
    assert (
        get_criteria_counts(
            a_values, b_values, scales, indifference_thresholds, preference_thresholds
        )
        == expected
    )


@pytest.mark.parametrize(
    (
        "alternatives",
        "veto_thresholds",
        "expected",
    ),
    (
        (
            [
                [90, 4, 600],
                [58, 0, 200],
                [66, 7, 400],
                [74, 8, 800],
                [98, 6, 800],
            ],
            [28, 8, 600],
            [
                [1.0, 0.2, 0.0, 0.2, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 1.0, 0.6, 0.0],
                [0.0, 0.2, 0.0, 1.0, 0.0],
                [0.0, 0.2, 0.0, 0.0, 1.0],
            ],
        ),
    ),
)
def test_credibility_electre_iv(
    alternatives,
    scales,
    indifference_thresholds,
    preference_thresholds,
    veto_thresholds, expected
) -> None:
    assert (
        credibility_electre_iv(
            alternatives,
            scales,
            indifference_thresholds,
            preference_thresholds,
            veto_thresholds,
        )
        == expected
    )
