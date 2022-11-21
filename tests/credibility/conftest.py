from typing import List

import pandas as pd
import pytest


@pytest.fixture
def alternative_names() -> List[str]:
    return [f"A{i}" for i in range(1, 9)]


@pytest.fixture
def profile_names() -> List[str]:
    return [f"B{i}" for i in range(1, 5)]


@pytest.fixture
def concordance_comprehensive(
    alternative_names: List[str], profile_names: List[str]
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1.0, 1.0, 0.846153846154, 0.615384615385],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.846153846154, 0.846153846154, 0.846153846154],
            [1.0, 1.0, 0.846153846154, 0.846153846154],
            [0.846153846154, 0.846153846154, 0.570850202429, 0.380566801619],
            [1.0, 0.692307692308, 0.570850202429, 0.307692307692],
            [0.846153846154, 0.230769230769, 0.230769230769, 0.230769230769],
            [1.0, 1.0, 0.878542510121, 0.615384615385],
        ],
        index=alternative_names,
        columns=profile_names,
    )


@pytest.fixture
def discordance_marginals(
    alternative_names: List[str], profile_names: List[str]
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            [
                pd.Series([0, 0, 0, 0, 0]),
                pd.Series([0, 0, 0, 0, 0]),
                pd.Series([0, 0, 1, 0, 0]),
                pd.Series([0, 0, 1, 1, 0]),
            ],
            [
                pd.Series([0, 0, 0, 0, 0]),
                pd.Series([0, 0, 0, 0, 0]),
                pd.Series([0, 0, 0, 0, 0]),
                pd.Series([0, 0, 0, 0, 0]),
            ],
            [
                pd.Series([0, 0, 0, 0, 0]),
                pd.Series([0, 0, 1, 0, 0]),
                pd.Series([0, 0, 1, 0, 0]),
                pd.Series([0, 0, 1, 0, 0]),
            ],
            [
                pd.Series([0, 0, 0, 0, 0]),
                pd.Series([0, 0, 0, 0, 0]),
                pd.Series([0, 0, 1, 0, 0]),
                pd.Series([0, 0, 1, 0, 0]),
            ],
            [
                pd.Series([1, 0, 0, 0, 0]),
                pd.Series([1, 0, 0, 0, 0]),
                pd.Series([1, 0, 1, 0, 0]),
                pd.Series([1, 0, 1, 1, 0]),
            ],
            [
                pd.Series([0, 0, 0, 0, 0]),
                pd.Series([0, 1, 1, 0, 0]),
                pd.Series([0, 1, 1, 0, 0]),
                pd.Series([0, 1, 1, 1, 0]),
            ],
            [
                pd.Series([1, 0, 0, 0, 0]),
                pd.Series([1, 1, 1, 0, 0.526315789474]),
                pd.Series([1, 1, 1, 0, 1]),
                pd.Series([1, 1, 1, 0, 1]),
            ],
            [
                pd.Series([0, 0, 0, 0, 0]),
                pd.Series([1, 0, 0, 0, 0]),
                pd.Series([0, 0, 0, 0, 0]),
                pd.Series([1, 0, 0, 1, 0]),
            ],
        ],
        index=alternative_names,
        columns=profile_names,
    )
