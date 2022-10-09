import math
from typing import List

import numpy as np
import pandas as pd
import pytest
from mcda.core.functions import Threshold
from mcda.electre.concordance import concordance, concordance_reinforced


@pytest.fixture
def reinforcement_thresholds(
    criterion_names: List[str],
) -> pd.Series:
    """Returns all criteria reinforcement thresholds."""
    return pd.Series(
        [
            Threshold(0, 3000),
            Threshold(0.25, 250),
            Threshold(0, 0.99),
            Threshold(0.2, 0),
            Threshold(0, 25),
        ],
        index=criterion_names,
    )


@pytest.fixture
def reinforcement_factors(
    criterion_names: List[str],
) -> pd.Series:
    """Returns all criteria reinforcement factors."""
    return pd.Series([1.5, 1.1, 1.2, 1.25, 1.3], index=criterion_names)


@pytest.fixture
def interactions(criterion_names: List[str]) -> pd.DataFrame:
    """Returns matrix with all interactions between criteria."""
    # TODO


def test_concordance(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    weights: pd.Series,
    indifference_thresholds: pd.Series,
    preference_thresholds: pd.Series,
) -> None:
    expected_values: List[List[float]] = [
        [
            1.0,
            0.307692307692,
            0.538461538462,
            1.0,
            0.307692307692,
            0.846153846154,
        ],
        [
            0.692307692308,
            1.0,
            0.692307692308,
            0.692307692308,
            1.0,
            0.923076923077,
        ],
        [
            1.0,
            0.307692307692,
            1.0,
            1.0,
            0.307692307692,
            1.0,
        ],
        [
            0.461538461538,
            0.307692307692,
            0.461538461538,
            1.0,
            0.307692307692,
            0.538461538462,
        ],
        [
            0.692307692308,
            0.692307692308,
            0.692307692308,
            0.692307692308,
            1.0,
            0.692307692308,
        ],
        [
            0.461538461538,
            0.0769230769231,
            0.0839160839161,
            0.461538461538,
            0.307692307692,
            1.0,
        ],
    ]

    result = concordance(
        performance_table,
        scales,
        weights,
        indifference_thresholds,
        preference_thresholds,
    )

    assert result.index.equals(performance_table.index)
    assert result.columns.equals(performance_table.index)
    result_array: np.ndarray = result.to_numpy()
    for expected_values_row, result_array_row in zip(expected_values, result_array):
        for expected_value, result_array_value in zip(
            expected_values_row, result_array_row
        ):
            assert math.isclose(expected_value, result_array_value)


def test_concordance_profiles(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    weights: pd.Series,
    indifference_thresholds: pd.Series,
    preference_thresholds: pd.Series,
    profiles_performance: pd.DataFrame,
) -> None:
    ...


def test_concordance_reinforced(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    weights: pd.Series,
    indifference_thresholds: pd.Series,
    preference_thresholds: pd.Series,
    reinforcement_thresholds: pd.Series,
    reinforcement_factors: pd.Series,
) -> None:
    expected_values: List[List[float]] = [
        [1.0, 0.383561643836, 0.538461538462, 1.0, 0.383561643836, 0.870967741935],
        [0.738562091503, 1.0, 0.701492537313, 0.738562091503, 1.0, 0.940476190476],
        [1.0, 0.383561643836, 1.0, 1.0, 0.383561643836, 1.0],
        [
            0.461538461538,
            0.383561643836,
            0.461538461538,
            1.0,
            0.383561643836,
            0.612903225806,
        ],
        [
            0.738562091503,
            0.692307692308,
            0.701492537313,
            0.738562091503,
            1.0,
            0.738562091503,
        ],
        [
            0.477611940299,
            0.0839694656489,
            0.0839160839161,
            0.513888888889,
            0.383561643836,
            1.0,
        ],
    ]

    result = concordance_reinforced(
        performance_table,
        scales,
        weights,
        indifference_thresholds,
        preference_thresholds,
        reinforcement_thresholds,
        reinforcement_factors,
    )

    print(result)

    assert result.index.equals(performance_table.index)
    assert result.columns.equals(performance_table.index)
    result_array: np.ndarray = result.to_numpy()

    for expected_values_row, result_array_row in zip(expected_values, result_array):
        for expected_value, result_array_value in zip(
            expected_values_row, result_array_row
        ):
            assert math.isclose(expected_value, result_array_value)


def test_concordance_reinforced_profiles(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    weights: pd.Series,
    indifference_thresholds: pd.Series,
    preference_thresholds: pd.Series,
    reinforcement_thresholds: pd.Series,
    reinforcement_factors: pd.Series,
    profiles_performance: pd.DataFrame,
) -> None:
    ...


# TODO: concordance with interactions
