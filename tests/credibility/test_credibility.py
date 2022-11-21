from typing import List

import pandas as pd
import pytest

from mcda.core.functions import Threshold
from mcda.core.scales import PreferenceDirection, QuantitativeScale
from mcda.electre.credibility import (
    credibility_comprehensive,
    credibility_cv,
    credibility_electre_iv,
    get_criteria_counts,
)
from mcda.electre.discordance import NonDiscordanceType, non_discordance

from .. import helpers


def test_credibility_comprehensive(
    concordance_comprehensive: pd.DataFrame, discordance_marginals: pd.DataFrame
) -> None:
    expected_values = [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.878542510121, 0.0],
    ]

    credibility_matrix: pd.DataFrame = credibility_comprehensive(
        concordance_comprehensive,
        non_discordance(
            discordance_marginals, NonDiscordanceType.DC, concordance_comprehensive
        ),
    )

    assert credibility_matrix.index.equals(concordance_comprehensive.index)
    assert credibility_matrix.columns.equals(concordance_comprehensive.columns)

    helpers.assert_array_values(expected_values, credibility_matrix.to_numpy())


@pytest.fixture
def counter_veto_occurs(
    alternative_names: List[str], profile_names: List[str]
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0, 0, 0, 1],
            [0, 1, 2, 3],
            [0, 1, 1, 2],
            [0, 1, 2, 2],
            [0, 0, 2, 2],
            [0, 2, 2, 3],
            [0, 1, 2, 2],
            [0, 2, 2, 3],
        ],
        index=alternative_names,
        columns=profile_names,
    )


def test_credibility_cv(
    concordance_comprehensive: pd.DataFrame,
    discordance_marginals: pd.DataFrame,
    counter_veto_occurs: pd.DataFrame,
) -> None:
    expected_values = [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.307692307692],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.878542510121, 0.615384615385],
    ]

    credibility_matrix: pd.DataFrame = credibility_cv(
        concordance_comprehensive,
        non_discordance(
            discordance_marginals, NonDiscordanceType.DC, concordance_comprehensive
        ),
        counter_veto_occurs,
        number_of_criteria=3,
    )

    assert credibility_matrix.index.equals(concordance_comprehensive.index)
    assert credibility_matrix.columns.equals(concordance_comprehensive.columns)

    helpers.assert_array_values(expected_values, credibility_matrix.to_numpy())


@pytest.fixture
def alt_names() -> List[str]:
    return ["Audi A3", "Audi A4", "BMW 118", "BMW 320", "Volvo C30", "Volvo S40"]


@pytest.fixture
def criteria_names() -> List[str]:
    return ["Price", "Power", "0-100", "Consumption", "CO2"]


@pytest.fixture
def performance_table(alt_names: List[str], criteria_names: List[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            [22080.0, 105.0, 11.40, 5.8, 119.0],
            [28100.0, 160.0, 8.6, 9.6, 164.0],
            [24650.0, 143.0, 9.0, 4.5, 119.0],
            [32700.0, 177.0, 7.9, 6.7, 128.0],
            [22750.0, 136.0, 9.4, 7.6, 151.0],
            [27350.0, 180.0, 7.9, 8.4, 164.0],
        ],
        index=alt_names,
        columns=criteria_names,
    )


@pytest.fixture
def scales(criteria_names: List[str]) -> pd.Series:
    return pd.Series(
        [
            QuantitativeScale(0, 40000, PreferenceDirection.MIN),
            QuantitativeScale(0, 200),
            QuantitativeScale(0, 15, PreferenceDirection.MIN),
            QuantitativeScale(0, 10, PreferenceDirection.MIN),
            QuantitativeScale(0, 200, PreferenceDirection.MIN),
        ],
        index=criteria_names,
    )


@pytest.fixture
def indifference_thresholds(criteria_names: List[str]) -> pd.Series:
    return pd.Series(
        [
            Threshold(0, 500),
            Threshold(0, 0),
            Threshold(0, 0),
            Threshold(0, 0),
            Threshold(0, 0),
        ],
        index=criteria_names,
    )


@pytest.fixture
def preference_thresholds(criteria_names: List[str]) -> pd.Series:
    return pd.Series(
        [
            Threshold(0, 3000),
            Threshold(0, 30),
            Threshold(0, 2),
            Threshold(0, 1),
            Threshold(0, 100),
        ],
        index=criteria_names,
    )


@pytest.fixture
def veto_thresholds(criteria_names: List[str]) -> pd.Series:
    return pd.Series(
        [
            Threshold(0, 4000),
            None,
            None,
            None,
            None,
        ],
        index=criteria_names,
    )


def test_get_criteria_counts_alternatives(
    alt_names: List[str],
    performance_table: pd.DataFrame,
    scales: pd.Series,
    indifference_thresholds: pd.Series,
    preference_thresholds: pd.Series,
) -> None:
    expected_values = pd.DataFrame(
        [
            [
                pd.Series([0, 0, 0, 5]),
                pd.Series([2, 1, 0, 0]),
                pd.Series([0, 1, 0, 1]),
                pd.Series([1, 2, 0, 0]),
                pd.Series([1, 2, 0, 0]),
                pd.Series([2, 1, 0, 0]),
            ],
            [
                pd.Series([2, 0, 0, 0]),
                pd.Series([0, 0, 0, 5]),
                pd.Series([0, 2, 0, 0]),
                pd.Series([1, 0, 0, 0]),
                pd.Series([0, 2, 0, 0]),
                pd.Series([0, 0, 0, 1]),
            ],
            [
                pd.Series([3, 0, 0, 1]),
                pd.Series([2, 1, 0, 0]),
                pd.Series([0, 0, 0, 5]),
                pd.Series([2, 1, 0, 0]),
                pd.Series([1, 3, 0, 0]),
                pd.Series([1, 2, 0, 0]),
            ],
            [
                pd.Series([2, 0, 0, 0]),
                pd.Series([1, 3, 0, 0]),
                pd.Series([1, 1, 0, 0]),
                pd.Series([0, 0, 0, 5]),
                pd.Series([1, 3, 0, 0]),
                pd.Series([1, 1, 0, 1]),
            ],
            [
                pd.Series([2, 0, 0, 0]),
                pd.Series([2, 1, 0, 0]),
                pd.Series([0, 1, 0, 0]),
                pd.Series([1, 0, 0, 0]),
                pd.Series([0, 0, 0, 5]),
                pd.Series([1, 2, 0, 0]),
            ],
            [
                pd.Series([2, 0, 0, 0]),
                pd.Series([1, 3, 0, 1]),
                pd.Series([1, 1, 0, 0]),
                pd.Series([1, 1, 0, 1]),
                pd.Series([1, 1, 0, 0]),
                pd.Series([0, 0, 0, 5]),
            ],
        ],
        index=alt_names,
        columns=alt_names,
    )

    criteria_counts_matrix: pd.DataFrame = get_criteria_counts(
        performance_table, scales, indifference_thresholds, preference_thresholds
    )

    pd.testing.assert_frame_equal(criteria_counts_matrix, expected_values)


@pytest.fixture
def prof_names() -> List[str]:
    return ["Bad", "Good"]


@pytest.fixture
def profiles_performance_table(
    prof_names: List[str], criteria_names: List[str]
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            [30000.0, 100.0, 11.0, 8.0, 125.0],
            [23000.0, 160.0, 8.0, 7.0, 120.0],
        ],
        index=prof_names,
        columns=criteria_names,
    )


def test_get_criteria_counts_alternatives_profiles(
    alt_names: List[str],
    performance_table: pd.DataFrame,
    scales: pd.Series,
    indifference_thresholds: pd.Series,
    preference_thresholds: pd.Series,
    prof_names: List[str],
    profiles_performance_table: pd.DataFrame,
) -> None:
    expected_values = pd.DataFrame(
        [
            [pd.Series([2, 2, 0, 0]), pd.Series([1, 2, 0, 0])],
            [pd.Series([2, 1, 0, 0]), pd.Series([0, 0, 0, 1])],
            [pd.Series([4, 1, 0, 0]), pd.Series([1, 1, 0, 0])],
            [pd.Series([3, 0, 0, 0]), pd.Series([0, 3, 0, 0])],
            [pd.Series([2, 2, 0, 0]), pd.Series([0, 0, 1, 0])],
            [pd.Series([2, 1, 0, 0]), pd.Series([0, 2, 0, 0])],
        ],
        index=alt_names,
        columns=prof_names,
    )

    criteria_counts_matrix: pd.DataFrame = get_criteria_counts(
        performance_table,
        scales,
        indifference_thresholds,
        preference_thresholds,
        profiles_performance_table,
    )

    pd.testing.assert_frame_equal(criteria_counts_matrix, expected_values)


def test_get_criteria_counts_profiles_alternatives(
    alt_names: List[str],
    performance_table: pd.DataFrame,
    scales: pd.Series,
    indifference_thresholds: pd.Series,
    preference_thresholds: pd.Series,
    prof_names: List[str],
    profiles_performance_table: pd.DataFrame,
) -> None:
    expected_values = pd.DataFrame(
        [
            [
                pd.Series([0, 1, 0, 0]),
                pd.Series([1, 1, 0, 0]),
                pd.Series([0, 0, 0, 0]),
                pd.Series([0, 2, 0, 0]),
                pd.Series([0, 1, 0, 0]),
                pd.Series([0, 2, 0, 0]),
            ],
            [
                pd.Series([2, 0, 0, 0]),
                pd.Series([2, 2, 0, 1]),
                pd.Series([0, 3, 0, 0]),
                pd.Series([1, 1, 0, 0]),
                pd.Series([0, 4, 0, 0]),
                pd.Series([2, 1, 0, 0]),
            ],
        ],
        index=prof_names,
        columns=alt_names,
    )

    criteria_counts_matrix: pd.DataFrame = get_criteria_counts(
        profiles_performance_table,
        scales,
        indifference_thresholds,
        preference_thresholds,
        performance_table,
    )

    pd.testing.assert_frame_equal(criteria_counts_matrix, expected_values)


def test_credibility_electre_iv_no_profiles(
    alt_names: List[str],
    performance_table: pd.DataFrame,
    scales: pd.Series,
    indifference_thresholds: pd.Series,
    preference_thresholds: pd.Series,
    veto_thresholds: pd.Series,
) -> None:
    expected_values = pd.DataFrame(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.8, 1.0, 0.2, 0.8, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ],
        index=alt_names,
        columns=alt_names,
    )

    credibility_matrix = credibility_electre_iv(
        performance_table,
        scales,
        indifference_thresholds,
        preference_thresholds,
        veto_thresholds,
    )

    pd.testing.assert_frame_equal(pd.DataFrame(credibility_matrix), expected_values)


def test_credibility_electre_iv_with_profiles(
    alt_names: List[str],
    performance_table: pd.DataFrame,
    scales: pd.Series,
    indifference_thresholds: pd.Series,
    preference_thresholds: pd.Series,
    veto_thresholds: pd.Series,
    prof_names: List[str],
    profiles_performance_table: pd.DataFrame,
) -> None:
    expected_alt_prof = pd.DataFrame(
        [
            [0.8, 0.0],
            [0.2, 0.0],
            [1.0, 0.4],
            [0.8, 0.0],
            [0.8, 0.4],
            [0.8, 0.0],
        ],
        index=alt_names,
        columns=prof_names,
    )

    expected_prof_alt = pd.DataFrame(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2, 1.0, 0.0, 0.4, 1.0, 0.8]],
        index=prof_names,
        columns=alt_names,
    )

    credibility_matrix_alt_prof, credibility_matrix_prof_alt = credibility_electre_iv(
        performance_table,
        scales,
        indifference_thresholds,
        preference_thresholds,
        veto_thresholds,
        profiles_performance_table,
    )

    pd.testing.assert_frame_equal(credibility_matrix_alt_prof, expected_alt_prof)  # type: ignore
    pd.testing.assert_frame_equal(credibility_matrix_prof_alt, expected_prof_alt)  # type: ignore
