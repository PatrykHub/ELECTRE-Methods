import copy

import pytest
from mcda.core.scales import QuantitativeScale
from mcda.electre._validate import (_all_lens_equal, _both_values_in_scale,
                                    _inverse_values,
                                    _reinforcement_factors_vals,
                                    _weights_proper_vals)


@pytest.mark.parametrize(
    ("aval", "bval", "scale", "expected", "match"),
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
def test_both_in_scale(aval, bval, scale, expected, match: str) -> None:
    if expected is None:
        _both_values_in_scale(aval, bval, scale)
    else:
        with pytest.raises(expected, match=match):
            _both_values_in_scale(aval, bval, scale)


@pytest.mark.parametrize(
    ("kwargs", "expected", "match"),
    (
        [{"a": [2, 3, 4], "b": [3, 4, 5], "c": ["a", "b", "c"]}, None, ""],
        [{"qqq": "abc", "qqqq": "XDD", "nice_variable": "OMG", "XD": "WOW"}, None, ""],
        [{":c": (3, 4), "arg2": [3, 4], "just_str": "ff"}, None, ""],
        [{"factors": [3, 4, 5, 1000]}, None, ""],
        [{"a1": [], "123": "", "": "", "--": ""}, None, ""],
        [
            {"a": [1, 1, 1, 1], "b": [2, 3], "c": [2, 3, 1, 2]},
            ValueError,
            " len\\(b\\)=2, len\\(a\\)=4",
        ],
        [
            {"1": (3, 4, 4), "2": [2, 3, 6], "3": [2, 3, 1, 2]},
            ValueError,
            " len\\(3\\)=4, len\\(1\\)=3",
        ],
        [{"factors": 3}, TypeError, "got 'int' instead."],
        [{"2333": 3, "xdd": []}, TypeError, " got 'int' instead"],
        [{"q": [], "w": 5, "e": 6}, TypeError, " got 'int' instead"],
    ),
)
def test_all_lens_equal(kwargs, expected, match: str) -> None:
    if expected is None:
        _all_lens_equal(**kwargs)
    else:
        with pytest.raises(expected, match=match):
            _all_lens_equal(**kwargs)


@pytest.mark.parametrize(
    ("aval", "bval", "scale", "inverse", "expected", "match"),
    (
        [3, 4, QuantitativeScale(0, 10), False, None, ""],
        [0, 10, QuantitativeScale(0, 10), True, None, ""],
    ),
)
def test_inverse(aval, bval, scale, inverse, expected, match: str) -> None:
    if expected is None:
        a, b, c = _inverse_values(aval, bval, copy.copy(scale), inverse)
        if not inverse:
            assert (aval, bval) == (a, b)
            assert c.preference_direction == scale.preference_direction
            assert c.dmin == scale.dmin
            assert c.dmax == scale.dmax
        else:
            assert aval == b
            assert bval == a
            assert c.preference_direction != scale.preference_direction
            assert c.dmin == scale.dmin
            assert c.dmax == scale.dmax
    else:
        with pytest.raises(expected, match=match):
            _inverse_values(aval, bval, scale, inverse)


@pytest.mark.parametrize(
    ("weights", "expected", "match"),
    [
        [(2, 3, 4), None, ""],
        [[0.00000001, 0.00000002, 9999], None, ""],
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
    [[(1.00000000001, 1.0, 99), None, ""], [[2, 3, 5], None, ""]],
)
def test_reinforcement_factors(factors, expected, match: str) -> None:
    if expected is None:
        _reinforcement_factors_vals(factors)
    else:
        with pytest.raises(expected, match=match):
            _reinforcement_factors_vals(factors)
