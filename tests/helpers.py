import math
from typing import List

import numpy as np


def assert_array_values(expected_values: List[List[float]], result_values: np.ndarray) -> None:
    """Checks if all values inside returned `pd.DataFrame` are correct."""
    for expected_row, result_row in zip(expected_values, result_values):
        for expected_value, result_value in zip(expected_row, result_row):
            assert math.isclose(expected_value, result_value)


def assert_cv_criteria_names(
    expected_values: List[List[List[str]]], result_values: np.ndarray
) -> None:
    """Checks if all lists with criteria names inside returned `pd.DataFrame` are correct"""
    for expected_row, result_row in zip(expected_values, result_values):
        for expected_value, result_value in zip(expected_row, result_row):
            assert sorted(expected_value) == sorted(result_value)
