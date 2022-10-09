import math
from typing import List

import pandas as pd
import pytest
from mcda.core.functions import Threshold
from mcda.core.scales import PreferenceDirection, QuantitativeScale


@pytest.fixture
def profile_names() -> List[str]:
    """Returns all profile names."""
    return ["P1", "P2", "P3"]


@pytest.fixture
def alternative_names() -> List[str]:
    """Returns all alternative names."""
    return ["A1", "A2", "A3", "A4", "A5", "A6"]


@pytest.fixture
def criterion_names() -> List[str]:
    """Returns all criterion names."""
    return ["Price", "Repair cost", "Rating", "Max speed", "Comfort"]


@pytest.fixture
def scales(criterion_names: List[str]) -> pd.Series:
    """Returns all criteria scales with provided preference direction."""
    return pd.Series(
        [
            QuantitativeScale(0, math.inf, PreferenceDirection.MIN),
            QuantitativeScale(0, math.inf, PreferenceDirection.MIN),
            QuantitativeScale(0, 10),
            QuantitativeScale(0, 300),
            QuantitativeScale(0, 100),
        ],
        index=criterion_names,
    )


@pytest.fixture
def performance_table(
    alternative_names: List[str],
    criterion_names: List[str],
) -> pd.DataFrame:
    """Returns performance table of all alternatives."""
    return pd.DataFrame(
        [
            [500, 250.50, 4.5, 200, 60],
            [10000, 30000, 9.99, 300, 98],
            [2019.5, 500, 6.63, 250, 75],
            [100, 1000, 3.33, 100, 65],
            [900000, 70000, 10, 300, 100],
            [20000, 3000, 5.5, 220, 30],
        ],
        index=alternative_names,
        columns=criterion_names,
    )


@pytest.fixture
def profiles_performance(
    profile_names: List[str],
    criterion_names: List[str],
) -> pd.DataFrame:
    """Returns performance table of all profiles."""
    return pd.DataFrame(
        [
            [3000, 1000, 3.33, 180, 40],
            [10000, 4000, 6.5, 220, 70],
            [50000, 5000, 9, 270, 85],
        ],
        index=profile_names,
        columns=criterion_names,
    )


@pytest.fixture
def weights(criterion_names: List[str]) -> pd.Series:
    """Returns all criteria weights."""
    return pd.Series([3, 1, 2, 4, 3], index=criterion_names)


@pytest.fixture
def indifference_thresholds(criterion_names: List[str]) -> pd.Series:
    """Returns all criteria indifference thresholds."""
    return pd.Series(
        [
            Threshold(0, 2000),
            Threshold(0.2, 150),
            Threshold(0, 0.75),
            Threshold(0.1, 0),
            Threshold(0, 15),
        ],
        index=criterion_names,
    )


@pytest.fixture
def preference_thresholds(criterion_names: List[str]) -> pd.Series:
    """Returns all criteria preference thresholds."""
    return pd.Series(
        [
            Threshold(0, 2500),
            Threshold(0.25, 150),
            Threshold(0, 0.95),
            Threshold(0.15, 0),
            Threshold(0, 20),
        ],
        index=criterion_names,
    )
