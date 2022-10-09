import math
from typing import List, Literal

import pandas as pd
import pytest
from mcda.core.functions import Threshold
from mcda.electre.discordance import discordance, discordance_bin


@pytest.fixture
def veto_thresholds(criterion_names: List[str]) -> pd.Series:
    """Returns all criteria veto thresholds."""
    return pd.Series(
        [
            Threshold(0.05, 10000),
            Threshold(0.4, 300),
            Threshold(0, 2.9),
            Threshold(0, 75),
            Threshold(0, 35),
        ],
        index=criterion_names,
    )


@pytest.fixture
def pre_veto_thresholds(criterion_names: List[str]) -> pd.Series:
    """Returns all criteria pre-veto thresholds."""
    return pd.Series(
        [
            Threshold(0.01, 6500),
            Threshold(0.3, 170),
            Threshold(0, 2.55),
            Threshold(0, 70),
            Threshold(0, 30),
        ],
        index=criterion_names,
    )


def test_discordance_bin(
    performance_table: pd.DataFrame, scales: pd.Series, veto_thresholds: pd.Series
) -> None:
    expected_values: List[List[Literal[0, 1]]] = [
        [0, 1, 0, 0, 1, 0],
        [1, 0, 1, 1, 0, 1],
        [0, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 0],
    ]

    result = discordance_bin(performance_table, scales, veto_thresholds)

    assert result.index.equals(performance_table.index)
    assert result.columns.equals(performance_table.index)

    result_matrix = result.to_numpy()
    for row_expected, row_result in zip(expected_values, result_matrix):
        for expected_value, result_value in zip(row_expected, row_result):
            assert int(result_value) == result_value
            assert expected_value == result_value


def test_discordance_bin_profiles(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    veto_thresholds: pd.Series,
    profiles_performance: pd.DataFrame,
) -> None:
    ...


def test_discordance(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    weights: pd.Series,
    preference_thresholds: pd.Series,
    veto_thresholds: pd.Series,
) -> None:
    expected_values: List[List[float]] = [
        [0.0, 0.692307692308, 0.229848783695, 0.0, 0.692307692308, 0.003944770317554],
        [0.278846153846, 0.0, 0.235000288462, 0.290384615385, 0.0, 0.0769230769231],
        [0.0, 0.302564102564, 0.0, 0.0, 0.333333333333, 0.0],
        [
            0.401972386588,
            0.661538461539,
            0.487179487179,
            0.0,
            0.692307692308,
            0.403944773176,
        ],
        [
            0.307692307692,
            0.307692307692,
            0.307692307692,
            0.307692307692,
            0.0,
            0.307692307692,
        ],
        [
            0.461538461539,
            0.89592760181,
            0.552662721893,
            0.538461538462,
            0.692307692308,
            0.0,
        ],
    ]

    result = discordance(
        performance_table, scales, weights, preference_thresholds, veto_thresholds
    )

    assert result.index.equals(performance_table.index)
    assert result.columns.equals(performance_table.index)

    result_matrix = result.to_numpy()
    for row_expected, row_result in zip(expected_values, result_matrix):
        for expected_value, result_value in zip(row_expected, row_result):
            assert math.isclose(expected_value, result_value)


def test_discordance_profiles(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    weights: pd.Series,
    preference_thresholds: pd.Series,
    veto_thresholds: pd.Series,
    profiles_performance: pd.DataFrame,
) -> None:
    ...


def test_discordance_pv(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    weights: pd.Series,
    preference_thresholds: pd.Series,
    veto_thresholds: pd.Series,
    pre_veto_thresholds: pd.Series,
) -> None:
    ...


def test_discordance_profiles_pv(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    weights: pd.Series,
    preference_thresholds: pd.Series,
    veto_thresholds: pd.Series,
    pre_veto_thresholds: pd.Series,
    profiles_performance: pd.DataFrame,
) -> None:
    ...
