import math
from typing import List

import pandas as pd
import pytest
from mcda.core.functions import Threshold
from mcda.core.scales import PreferenceDirection, QuantitativeScale


@pytest.fixture
def profile_names() -> List[str]:
    """Returns all profile names."""
    return ["P1", "P2", "P3", "P4", "P5"]


@pytest.fixture
def alternative_names() -> List[str]:
    """Returns all alternative names."""
    return [
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
        "A9",
        "B1",
        "B2",
        "B3",
        "B4",
    ]


@pytest.fixture
def criterion_names() -> List[str]:
    """Returns all criterion names."""
    return [
        "Price",
        "Repair cost",
        "Rating",
        "Max speed",
        "Comfort",
        "Appearance",
        "Weight",
        "Latency time",
    ]


@pytest.fixture
def scales(criterion_names: List[str]) -> pd.Series:
    """Returns all criteria scales with provided preference direction."""
    return pd.Series(
        [
            QuantitativeScale(0, math.inf, PreferenceDirection.MIN),
            QuantitativeScale(0, math.inf, PreferenceDirection.MIN),
            QuantitativeScale(0, 10),
            QuantitativeScale(80, 300),
            QuantitativeScale(0, 100),
            QuantitativeScale(0, 10),
            QuantitativeScale(300, 2000, PreferenceDirection.MIN),
            QuantitativeScale(0, 365, PreferenceDirection.MIN),
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
            [141770.84, 78688.95, 2.25, 81, 24.84, 4.53, 487, 80],
            [187603.7, 36951.14, 9.69, 199, 98.76, 8.22, 1088, 166],
            [134619.64, 169730.25, 8.8, 113, 62.38, 8.83, 612, 27],
            [108596.57, 66249.12, 2.42, 295, 51.13, 8.93, 1941, 239],
            [105957.1, 119678.87, 3.93, 266, 72.71, 5.74, 1338, 244],
            [46792.93, 173132.21, 1.01, 239, 90.23, 2.65, 963, 188],
            [194107.13, 27180.7, 2.79, 182, 40.62, 4.72, 312, 25],
            [35436.44, 174192.78, 7.45, 289, 70.39, 5.35, 849, 350],
            [23758.84, 64186.47, 9.02, 99, 0.92, 9.7, 1417, 237],
            [45360.48, 72572.72, 0.94, 245, 26.56, 0.13, 1537, 240],
            [21695.33, 140455.66, 1.24, 261, 75.22, 3.06, 428, 360],
            [126165.2, 127474.45, 6.21, 110, 44.03, 4.98, 1235, 193],
            [5699.74, 107708.21, 5.5, 300, 80.77, 0.25, 1819, 288],
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
            [1924.84, 10352.3, 8.41, 237, 89.68, 9.48, 1403, 95],
            [110452.95, 14465.84, 7.31, 163, 33.66, 7.66, 1539, 149],
            [159197.62, 110629.27, 5.31, 147, 26.55, 6.96, 1707, 151],
            [176483.55, 120435.19, 2.05, 100, 24.07, 3.17, 1909, 231],
            [194200.5, 135381.56, 1.29, 89, 21.5, 1.63, 1994, 315],
        ],
        index=profile_names,
        columns=criterion_names,
    )


@pytest.fixture
def weights(criterion_names: List[str]) -> pd.Series:
    """Returns all criteria weights."""
    return pd.Series([3, 5, 2, 3, 4, 1.75, 4, 1], index=criterion_names)


@pytest.fixture
def indifference_thresholds(criterion_names: List[str]) -> pd.Series:
    """Returns all criteria indifference thresholds."""
    return pd.Series(
        [
            Threshold(0.1, 10000),
            Threshold(0.15, 7000),
            Threshold(0, 0.5),
            Threshold(0, 40),
            Threshold(0, 10),
            Threshold(0, 0.75),
            Threshold(0, 200),
            Threshold(0, 30),
        ],
        index=criterion_names,
    )


@pytest.fixture
def preference_thresholds(criterion_names: List[str]) -> pd.Series:
    """Returns all criteria preference thresholds."""
    return pd.Series(
        [
            Threshold(0.1, 30000),
            Threshold(0.2, 7000),
            Threshold(0, 1.5),
            Threshold(0, 70),
            Threshold(0, 15),
            Threshold(0, 1),
            Threshold(0, 300),
            Threshold(0, 60),
        ],
        index=criterion_names,
    )
