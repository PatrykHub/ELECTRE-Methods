import copy

import pandas as pd
import pytest

from mcda.core.scales import QuantitativeScale
from mcda.electre._validation import (
    _both_values_in_scale,
    _inverse_values,
    _reinforcement_factors_vals,
    _weights_proper_vals,
)


@pytest.mark.parametrize(
    ("a_value", "b_value", "scale", "expected", "match"),
    (
        [3, 10, QuantitativeScale(3, 10), None, ""],
        [-0.0, -50, QuantitativeScale(-100, 0), None, ""],
        [9.99999, 100, QuantitativeScale(10, 101), ValueError, " interval"],
        [10, 0.0000001, QuantitativeScale(0.0000002, 15), ValueError, " interval"],
        [10, 10, QuantitativeScale(0, 2), ValueError, " interval"],
        ["a", 0, QuantitativeScale(-10, 10), TypeError, "got \\('str', 'int'\\)"],
        [
            0,
            (3, 4),
            QuantitativeScale(0, 10),
            TypeError,
            "got \\('int', 'tuple'\\)",
        ],
        [0.0, 0.0, [0.0, 0.0], TypeError, "got 'list'"],
        [0.0, 0.0, [1, 2], TypeError, "got 'list'"],
        ["a", "b", "abc", TypeError, "got 'str'"],
        [1, 2, "aaaaa", TypeError, "got 'str'"],
    ),
)
def test_both_in_scale(a_value, b_value, scale, expected, match: str) -> None:
    if expected is None:
        _both_values_in_scale(a_value, b_value, scale)
    else:
        with pytest.raises(expected, match=match):
            _both_values_in_scale(a_value, b_value, scale)


@pytest.mark.parametrize(
    ("a_value", "b_value", "scale", "inverse", "expected", "match"),
    (
        [3, 4, QuantitativeScale(0, 10), False, None, ""],
        [0, 10, QuantitativeScale(0, 10), True, None, ""],
    ),
)
def test_inverse(a_value, b_value, scale, inverse, expected, match: str) -> None:
    if expected is None:
        a, b, c = _inverse_values(a_value, b_value, copy.copy(scale), inverse)
        if not inverse:
            assert (a_value, b_value) == (a, b)
            assert c.preference_direction == scale.preference_direction
            assert c.dmin == scale.dmin
            assert c.dmax == scale.dmax
        else:
            assert a_value == b
            assert b_value == a
            assert c.preference_direction != scale.preference_direction
            assert c.dmin == scale.dmin
            assert c.dmax == scale.dmax
    else:
        with pytest.raises(expected, match=match):
            _inverse_values(a_value, b_value, scale, inverse)


@pytest.mark.parametrize(
    ("weights", "expected", "match"),
    [
        [pd.Series([2, 3, 4], index=["2137", "ABC", "..."]), None, ""],
        [{"key1": 1.00000001, "abcdef": 44.00000002, "33": 9999}, None, ""],
    ],
)
def test_weights(weights, expected, match: str) -> None:
    if expected is None:
        _weights_proper_vals(weights)
    else:
        with pytest.raises(expected, match=match):
            _weights_proper_vals(weights)


@pytest.mark.parametrize(
    ("factors", "expected", "match"),
    [
        [pd.Series([1.00000000001, 1.4, 99], index=["2137", "ABC", "..."]), None, ""],
        [{"key a": 2, "qqq": 3, "ewq": 5}, None, ""],
    ],
)
def test_reinforcement_factors(factors, expected, match: str) -> None:
    if expected is None:
        _reinforcement_factors_vals(factors)
    else:
        with pytest.raises(expected, match=match):
            _reinforcement_factors_vals(factors)
