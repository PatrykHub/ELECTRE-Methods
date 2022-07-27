
from mcda.core.aliases import NumericValue
from mcda.core.functions import Threshold

import pytest


@pytest.mark.parametrize(
    ('alpha', 'beta', 'argument', 'expected'),
    (
        [2, 0, 0.5, 1.0],
        [0, 9, 100, 9]
    )
)
def test_simple_calc(
    alpha: NumericValue,
    beta: NumericValue,
    argument: NumericValue,
    expected: NumericValue
) -> None:
    assert Threshold(alpha, beta)(argument) == expected
