# type: ignore
from typing import List

import pandas as pd
import pytest

from mcda.core.functions import Threshold
from mcda.electre.discordance import (
    counter_veto,
    counter_veto_count,
    discordance,
    discordance_bin,
)

from .. import helpers


@pytest.fixture
def veto_thresholds(criterion_names: List[str]) -> pd.Series:
    """Returns all criteria veto thresholds."""
    return pd.Series(
        [
            Threshold(0.2, 40000),
            None,
            Threshold(0, 3),
            Threshold(0, 100),
            Threshold(0, 45),
            None,
            Threshold(0, 500),
            Threshold(0, 120),
        ],
        index=criterion_names,
    )


@pytest.fixture
def pre_veto_thresholds(criterion_names: List[str]) -> pd.Series:
    """Returns pre-veto thresholds."""
    return pd.Series(
        [
            Threshold(0.2, 30000),
            None,
            None,
            Threshold(0, 90),
            Threshold(0, 25),
            None,
            None,
            Threshold(0, 80),
        ],
        index=criterion_names,
    )


@pytest.fixture
def counter_veto_thresholds(criterion_names: List[str]) -> pd.Series:
    """Returns counter-veto thresholds."""
    return pd.Series(
        [
            Threshold(0.17, 21370),
            Threshold(0.1, 50000),
            Threshold(0, 4.5),
            Threshold(0, 105),
            Threshold(0, 49),
            None,
            None,
            Threshold(0, 110),
        ],
        index=criterion_names,
    )


def test_discordance_binary_no_profiles(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    veto_thresholds: pd.Series,
) -> None:
    expected_values = [
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0],
    ]

    discordance_matrix: pd.DataFrame = discordance_bin(performance_table, scales, veto_thresholds)

    assert discordance_matrix.index.equals(performance_table.index)
    assert discordance_matrix.columns.equals(performance_table.index)

    helpers.assert_array_values(expected_values, discordance_matrix.to_numpy())


def test_discordance_binary_profiles(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    veto_thresholds: pd.Series,
    profiles_performance: pd.DataFrame,
) -> None:
    expected_alternatives_profiles = [
        [1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0],
    ]
    expected_profiles_alternatives = [
        [1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]

    discordance_matrix_alt_prof, discordance_matrix_prof_alt = discordance_bin(
        performance_table,
        scales,
        veto_thresholds,
        profiles_performance,
    )

    assert discordance_matrix_alt_prof.index.equals(performance_table.index)
    assert discordance_matrix_alt_prof.columns.equals(profiles_performance.index)

    assert discordance_matrix_prof_alt.index.equals(profiles_performance.index)
    assert discordance_matrix_prof_alt.columns.equals(performance_table.index)

    helpers.assert_array_values(
        expected_alternatives_profiles, discordance_matrix_alt_prof.to_numpy()
    )
    helpers.assert_array_values(
        expected_profiles_alternatives, discordance_matrix_prof_alt.to_numpy()
    )


def test_discordance_no_profiles(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    preference_thresholds: pd.Series,
    veto_thresholds: pd.Series,
    weights: pd.Series,
) -> None:
    expected_values = [
        [
            0.0,
            0.378947368421,
            0.210750877193,
            0.189698245614,
            0.304842105263,
            0.421052631579,
            0.130694736842,
            0.505263157895,
            0.210526315789,
            0.252631578947,
            0.421052631579,
            0.107733333333,
            0.505263157895,
        ],
        [
            0.186666666667,
            0.0,
            0.208866271851,
            0.235789473684,
            0.126315789474,
            0.126315789474,
            0.210526315789,
            0.21052631579,
            0.126315789474,
            0.126315789474,
            0.294736842105,
            0.0556824547112,
            0.252631578947,
        ],
        [
            0.0,
            0.187396491228,
            0.0,
            0.126315789474,
            0.126315789474,
            0.324771929825,
            0.0,
            0.252631578947,
            0.126315789474,
            0.252631578947,
            0.252631578947,
            0.0,
            0.271663157895,
        ],
        [
            0.210526315789,
            0.430175438597,
            0.294736842105,
            0.0,
            0.205922807017,
            0.430035087719,
            0.210526315789,
            0.402863157895,
            0.378947368421,
            0.213894736842,
            0.345768421053,
            0.252631578947,
            0.292715789474,
        ],
        [
            0.210526315789,
            0.158877192982,
            0.294736842105,
            0.0,
            0.0,
            0.191187701987,
            0.210526315789,
            0.369684210526,
            0.210526315789,
            0.122667814649,
            0.294736842105,
            0.0437894736842,
            0.130245614035,
        ],
        [
            0.181894736842,
            0.0842105263158,
            0.169263157895,
            0.0,
            0.0797192982456,
            0.0,
            0.226245614035,
            0.0842105263158,
            0.0842105263158,
            0.0,
            0.168421052632,
            0.0842105263158,
            0.139402316762,
        ],
        [
            0.0125650326268,
            0.252631578947,
            0.165440056498,
            0.252631578947,
            0.281207017544,
            0.294736842105,
            0.0,
            0.419761403509,
            0.210526315789,
            0.126315789474,
            0.274245614035,
            0.163800072658,
            0.461754385965,
        ],
        [
            0.0943157894737,
            0.15870877193,
            0.0421052631579,
            0.0357894736842,
            0.0322807017544,
            0.0692771929824,
            0.210526315789,
            0.0,
            0.0411228070175,
            0.0350877192982,
            0.101894736842,
            0.0421052631579,
            0.00140350877193,
        ],
        [
            0.260603508772,
            0.326877192982,
            0.378947368421,
            0.294736842105,
            0.294736842105,
            0.424421052632,
            0.403929824561,
            0.463157894737,
            0.0,
            0.186049122807,
            0.463157894737,
            0.157810526316,
            0.294736842105,
        ],
        [
            0.210526315789,
            0.387929824561,
            0.411621052632,
            0.0537263157895,
            0.252070175439,
            0.336842105263,
            0.230175438596,
            0.414484210526,
            0.0842105263158,
            0.0,
            0.336842105263,
            0.0997614035088,
            0.297164282457,
        ],
        [
            0.0421052631579,
            0.174259649123,
            0.126315789474,
            0.0421052631579,
            0.106105263158,
            0.0421614035088,
            0.0449122807018,
            0.0842105263158,
            0.126315789474,
            0.0421052631579,
            0.0,
            0.126315789474,
            0.0926315789474,
        ],
        [
            0.205614035088,
            0.332631578947,
            0.29052631579,
            0.126315789474,
            0.203115789474,
            0.421052631579,
            0.218947368421,
            0.388828070175,
            0.199859649123,
            0.252631578947,
            0.511943859649,
            0.0,
            0.374680701754,
        ],
        [
            0.210526315789,
            0.311522807018,
            0.294736842105,
            0.0,
            0.152421052632,
            0.19649122807,
            0.210526315789,
            0.193684210526,
            0.170105263158,
            0.0,
            0.168421052632,
            0.19298245614,
            0.0,
        ],
    ]

    discordance_matrix: pd.DataFrame = discordance(
        performance_table, scales, weights, preference_thresholds, veto_thresholds
    )

    print(discordance_matrix)

    assert discordance_matrix.index.equals(performance_table.index)
    assert discordance_matrix.columns.equals(performance_table.index)

    helpers.assert_array_values(expected_values, discordance_matrix.to_numpy())


def test_discordance_profiles(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    preference_thresholds: pd.Series,
    veto_thresholds: pd.Series,
    profiles_performance: pd.DataFrame,
    weights: pd.Series,
) -> None:
    expected_alternatives_profiles = [
        [0.505263157895, 0.134736842105, 0.0842105263158, 0.0, 0.0],
        [0.134035087719, 0.124690790249, 0.0, 0.0, 0.0],
        [0.321684210526, 0.0, 0.0, 0.0, 0.0],
        [0.553263157895, 0.191157894737, 0.0976842105264, 0.0, 0.0],
        [0.26369122807, 0.108771929825, 0.0231578947368, 0.0, 0.0],
        [0.195043347689, 0.0842105263158, 0.0842105263158, 0.0, 0.0],
        [0.378947368421, 0.210526315789, 0.0572631578947, 0.0, 0.0],
        [0.0661894736842, 0.0421052631579, 0.0421052631579, 0.0414035087719, 0.0],
        [0.336842105263, 0.119242105263, 0.0779228070175, 0.045754385965, 0.0313263157895],
        [0.372072785786, 0.105964912281, 0.104561403509, 0.0, 0.0],
        [0.126315789474, 0.126315789474, 0.126315789474, 0.0421052631579, 0.0],
        [0.48701754386, 0.0, 0.0, 0.0, 0.0],
        [0.218947368421, 0.0595087719299, 0.0421052631579, 0.0, 0.0],
    ]
    expected_profiles_alternatives = [
        [
            0.168421052632,
            0.0126315789474,
            0.174035087719,
            0.0,
            0.0,
            0.117894736842,
            0.175438596491,
            0.168421052632,
            0.0,
            0.0,
            0.168421052632,
            0.0,
            0.0,
        ],
        [
            0.174736842105,
            0.34498245614,
            0.287550877193,
            0.14018245614,
            0.261333333333,
            0.488421052632,
            0.210526315789,
            0.543045614035,
            0.138105263158,
            0.176842105263,
            0.561740350877,
            0.00336842105263,
            0.421052631579,
        ],
        [
            0.176140350877,
            0.421052631579,
            0.411677192982,
            0.202911749437,
            0.388518641426,
            0.555789473684,
            0.210526315789,
            0.61889122807,
            0.210526315789,
            0.244210526316,
            0.589473684211,
            0.158764912281,
            0.421052631579,
        ],
        [
            0.210526315789,
            0.546666666667,
            0.4256,
            0.286484343657,
            0.589013346351,
            0.589473684211,
            0.269754385965,
            0.673684210526,
            0.372210526316,
            0.313263157895,
            0.589473684211,
            0.292675478425,
            0.505263157895,
        ],
        [
            0.223448158654,
            0.589473684211,
            0.483653785864,
            0.345992982456,
            0.661192982456,
            0.631578947368,
            0.330498245614,
            0.673684210526,
            0.391578947368,
            0.395368421053,
            0.589473684211,
            0.416935611419,
            0.505263157895,
        ],
    ]

    discordance_matrix_alt_prof, discordance_matrix_prof_alt = discordance(
        performance_table,
        scales,
        weights,
        preference_thresholds,
        veto_thresholds,
        profiles_perform=profiles_performance,
    )

    assert discordance_matrix_alt_prof.index.equals(performance_table.index)
    assert discordance_matrix_alt_prof.columns.equals(profiles_performance.index)

    assert discordance_matrix_prof_alt.index.equals(profiles_performance.index)
    assert discordance_matrix_prof_alt.columns.equals(performance_table.index)

    helpers.assert_array_values(
        expected_alternatives_profiles, discordance_matrix_alt_prof.to_numpy()
    )
    helpers.assert_array_values(
        expected_profiles_alternatives, discordance_matrix_prof_alt.to_numpy()
    )


def test_discordance_no_profiles_pre_veto(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    preference_thresholds: pd.Series,
    veto_thresholds: pd.Series,
    pre_veto_thresholds: pd.Series,
    weights: pd.Series,
) -> None:
    expected_values = [
        [
            0.0,
            0.378947368421,
            0.189810526316,
            0.137178947368,
            0.304842105263,
            0.421052631579,
            0.126315789474,
            0.505263157895,
            0.210526315789,
            0.252631578947,
            0.421052631579,
            0.0842105263158,
            0.505263157895,
        ],
        [
            0.174736842105,
            0.0,
            0.190315789474,
            0.202105263158,
            0.126315789474,
            0.126315789474,
            0.210526315789,
            0.126315789474,
            0.126315789474,
            0.126315789474,
            0.294736842105,
            0.0,
            0.252631578947,
        ],
        [
            0.0,
            0.0958315789474,
            0.0,
            0.126315789474,
            0.126315789474,
            0.276631578947,
            0.0,
            0.252631578947,
            0.126315789474,
            0.252631578947,
            0.252631578947,
            0.0,
            0.252631578947,
            0.272,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.210526315789,
            0.421052631579,
            0.294736842105,
            0.0,
            0.16898245614,
            0.413473684211,
            0.210526315789,
            0.378947368421,
            0.378947368421,
            0.213894736842,
            0.294736842105,
            0.252631578947,
            0.2496,
        ],
        [
            0.210526315789,
            0.0930526315789,
            0.294736842105,
            0.0,
            0.0,
            0.163866315789,
            0.210526315789,
            0.369684210526,
            0.210526315789,
            0.118802526316,
            0.294736842105,
            0.0437894736842,
            0.130245614035,
        ],
        [
            0.177684210526,
            0.0842105263158,
            0.169263157895,
            0.0,
            0.0797192982456,
            0.0,
            0.226245614035,
            0.0842105263158,
            0.0842105263158,
            0.0,
            0.168421052632,
            0.0842105263158,
            0.106121313684,
        ],
        [
            0.0,
            0.252631578947,
            0.0842105263158,
            0.252631578947,
            0.186021052632,
            0.294736842105,
            0.0,
            0.377010526316,
            0.210526315789,
            0.126315789474,
            0.207157894737,
            0.0842105263158,
            0.448140350877,
        ],
        [
            0.0943157894737,
            0.112028070175,
            0.0421052631579,
            0.0326315789474,
            0.0273684210526,
            0.0421052631579,
            0.210526315789,
            0.0,
            0.0386666666667,
            0.0315789473684,
            0.101894736842,
            0.0421052631579,
            0.0,
            0.0421052631579,
        ],
        [
            0.210526315789,
            0.319157894737,
            0.378947368421,
            0.294736842105,
            0.294736842105,
            0.424421052632,
            0.334315789474,
            0.463157894737,
            0.0,
            0.131705263158,
            0.463157894737,
            0.152505263158,
            0.294736842105,
        ],
        [
            0.210526315789,
            0.378105263158,
            0.385852631579,
            0.0,
            0.252070175439,
            0.336842105263,
            0.230175438596,
            0.4112,
            0.0842105263158,
            0.0,
            0.336842105263,
            0.0858947368421,
            0.260067082105,
        ],
        [
            0.0421052631579,
            0.126315789474,
            0.126315789474,
            0.0421052631579,
            0.104701754386,
            0.0421052631579,
            0.0449122807018,
            0.0842105263158,
            0.126315789474,
            0.0421052631579,
            0.0,
            0.126315789474,
            0.0842105263158,
        ],
        [
            0.203157894737,
            0.252631578947,
            0.271719298246,
            0.126315789474,
            0.157305263158,
            0.421052631579,
            0.210526315789,
            0.336505263158,
            0.199859649123,
            0.252631578947,
            0.473178947368,
            0.0,
            0.351494736842,
        ],
        [
            0.210526315789,
            0.294736842105,
            0.294736842105,
            0.0,
            0.152421052632,
            0.189473684211,
            0.210526315789,
            0.193684210526,
            0.170105263158,
            0.0,
            0.168421052632,
            0.184210526316,
            0.0,
        ],
    ]

    discordance_matrix: pd.DataFrame = discordance(
        performance_table,
        scales,
        weights,
        preference_thresholds,
        veto_thresholds,
        pre_veto_thresholds,
    )

    print(discordance_matrix)

    assert discordance_matrix.index.equals(performance_table.index)
    assert discordance_matrix.columns.equals(performance_table.index)

    helpers.assert_array_values(expected_values, discordance_matrix.to_numpy())


def test_discordance_profiles_pre_veto(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    preference_thresholds: pd.Series,
    veto_thresholds: pd.Series,
    pre_veto_thresholds: pd.Series,
    profiles_performance: pd.DataFrame,
    weights: pd.Series,
) -> None:
    expected_alternatives_profiles = [
        [0.505263157895, 0.0842105263158, 0.0842105263158, 0.0, 0.0],
        [0.126315789474, 0.121642231579, 0.0, 0.0, 0.0],
        [0.272, 0.0, 0.0, 0.0, 0.0],
        [0.535157894737, 0.180631578947, 0.0864561403509, 0.0, 0.0],
        [0.252631578947, 0.1, 0.0136842105263, 0.0, 0.0],
        [0.167488471579, 0.0842105263158, 0.0842105263158, 0.0, 0.0],
        [0.378947368421, 0.210526315789, 0.0572631578947, 0.0, 0.0],
        [0.0421052631579, 0.0421052631579, 0.0421052631579, 0.0410526315789, 0.0],
        [0.336842105263, 0.0736, 0.0116210526316, 0.0, 0.0],
        [0.349855292632, 0.0957894736842, 0.0936842105263, 0.0, 0.0],
        [0.126315789474, 0.126315789474, 0.126315789474, 0.0421052631579, 0.0],
        [0.479298245614, 0.0, 0.0, 0.0, 0.0],
        [0.218947368421, 0.0595087719299, 0.0421052631579, 0.0, 0.0],
    ]
    expected_profiles_alternatives = [
        [
            0.168421052632,
            0.0126315789474,
            0.168421052632,
            0.0,
            0.0,
            0.117894736842,
            0.168421052632,
            0.168421052632,
            0.0,
            0.0,
            0.168421052632,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.168421052632,
            0.34498245614,
            0.241852631579,
            0.126315789474,
            0.244631578947,
            0.463157894737,
            0.210526315789,
            0.519831578947,
            0.138105263158,
            0.126315789474,
            0.535242105263,
            0.00336842105263,
            0.421052631579,
        ],
        [
            0.168421052632,
            0.421052631579,
            0.385936842105,
            0.126315789474,
            0.352842105263,
            0.488421052632,
            0.210526315789,
            0.615635087719,
            0.210526315789,
            0.227368421053,
            0.589473684211,
            0.144842105263,
            0.421052631579,
        ],
        [
            0.210526315789,
            0.534736842105,
            0.406821052632,
            0.176382357895,
            0.550551101754,
            0.589473684211,
            0.210526315789,
            0.673684210526,
            0.372210526316,
            0.313263157895,
            0.589473684211,
            0.252631578947,
            0.505263157895,
        ],
        [
            0.210526315789,
            0.589473684211,
            0.428463157895,
            0.291621052632,
            0.653473684211,
            0.631578947368,
            0.248421052632,
            0.673684210526,
            0.378947368421,
            0.384842105263,
            0.589473684211,
            0.294736842105,
            0.505263157895,
        ],
    ]

    discordance_matrix_alt_prof, discordance_matrix_prof_alt = discordance(
        performance_table,
        scales,
        weights,
        preference_thresholds,
        veto_thresholds,
        pre_veto_thresholds,
        profiles_performance,
    )

    assert discordance_matrix_alt_prof.index.equals(performance_table.index)
    assert discordance_matrix_alt_prof.columns.equals(profiles_performance.index)

    assert discordance_matrix_prof_alt.index.equals(profiles_performance.index)
    assert discordance_matrix_prof_alt.columns.equals(performance_table.index)

    helpers.assert_array_values(
        expected_alternatives_profiles, discordance_matrix_alt_prof.to_numpy()
    )
    helpers.assert_array_values(
        expected_profiles_alternatives, discordance_matrix_prof_alt.to_numpy()
    )


def test_counter_veto_no_profiles(
    performance_table: pd.DataFrame, scales: pd.Series, counter_veto_thresholds: pd.Series
) -> None:
    expected_values = [
        [
            [],
            [],
            ["Repair cost"],
            ["Latency time"],
            ["Latency time"],
            ["Repair cost"],
            [],
            ["Latency time", "Repair cost"],
            ["Latency time"],
            ["Latency time"],
            ["Latency time"],
            ["Latency time"],
            ["Latency time"],
        ],
        [
            ["Comfort", "Max speed", "Rating"],
            [],
            ["Repair cost"],
            ["Rating"],
            ["Rating", "Repair cost"],
            ["Rating", "Repair cost"],
            ["Comfort", "Rating"],
            ["Latency time", "Repair cost"],
            ["Comfort"],
            ["Comfort", "Rating"],
            ["Latency time", "Rating", "Repair cost"],
            ["Comfort", "Repair cost"],
            ["Latency time", "Repair cost"],
        ],
        [
            ["Rating"],
            ["Latency time"],
            [],
            ["Latency time", "Rating"],
            ["Latency time", "Rating"],
            ["Latency time", "Rating"],
            ["Price", "Rating"],
            ["Latency time"],
            ["Comfort", "Latency time"],
            ["Latency time", "Rating"],
            ["Latency time", "Rating"],
            ["Latency time"],
            ["Latency time"],
        ],
        [
            ["Max speed"],
            ["Price"],
            ["Max speed", "Repair cost"],
            [],
            [],
            ["Repair cost"],
            ["Max speed", "Price"],
            ["Latency time", "Repair cost"],
            ["Comfort", "Max speed"],
            [],
            ["Latency time", "Repair cost"],
            ["Max speed"],
            [],
        ],
        [
            ["Max speed"],
            ["Price"],
            ["Max speed"],
            [],
            [],
            [],
            ["Price"],
            [],
            ["Comfort", "Max speed"],
            [],
            ["Latency time"],
            ["Max speed"],
            [],
        ],
        [
            ["Comfort", "Max speed", "Price"],
            ["Price"],
            ["Max speed", "Price"],
            ["Price"],
            ["Price"],
            [],
            ["Comfort", "Price"],
            ["Latency time"],
            ["Comfort", "Max speed"],
            ["Comfort"],
            ["Latency time"],
            ["Max speed", "Price"],
            [],
        ],
        [
            [],
            ["Latency time"],
            ["Repair cost"],
            ["Latency time"],
            ["Latency time", "Repair cost"],
            ["Latency time", "Repair cost"],
            [],
            ["Latency time", "Repair cost"],
            ["Latency time"],
            ["Latency time"],
            ["Latency time", "Repair cost"],
            ["Latency time", "Repair cost"],
            ["Latency time", "Repair cost"],
        ],
        [
            ["Max speed", "Price", "Rating"],
            ["Price"],
            ["Max speed", "Price"],
            ["Price", "Rating"],
            ["Price"],
            ["Rating"],
            ["Max speed", "Price", "Rating"],
            [],
            ["Comfort", "Max speed"],
            ["Rating"],
            ["Rating"],
            ["Max speed", "Price"],
            [],
        ],
        [
            ["Price", "Rating"],
            ["Price"],
            ["Price", "Repair cost"],
            ["Price", "Rating"],
            ["Price", "Rating"],
            ["Rating", "Repair cost"],
            ["Price", "Rating"],
            ["Latency time", "Repair cost"],
            [],
            ["Rating"],
            ["Latency time", "Rating", "Repair cost"],
            ["Price", "Repair cost"],
            [],
        ],
        [
            ["Max speed", "Price"],
            ["Price"],
            ["Max speed", "Price", "Repair cost"],
            ["Price"],
            ["Price"],
            ["Repair cost"],
            ["Price"],
            ["Repair cost"],
            ["Max speed"],
            [],
            ["Latency time", "Repair cost"],
            ["Max speed", "Price"],
            [],
        ],
        [
            ["Comfort", "Max speed", "Price"],
            ["Price"],
            ["Max speed", "Price"],
            ["Price"],
            ["Price"],
            [],
            ["Price"],
            [],
            ["Comfort", "Max speed"],
            [],
            [],
            ["Max speed", "Price"],
            [],
        ],
        [
            [],
            ["Price"],
            [],
            [],
            [],
            ["Rating"],
            ["Price"],
            ["Latency time"],
            [],
            ["Rating"],
            ["Latency time", "Rating"],
            [],
            [],
        ],
        [
            ["Comfort", "Max speed", "Price"],
            ["Price"],
            ["Max speed", "Price"],
            ["Price"],
            ["Price"],
            ["Price"],
            ["Max speed", "Price"],
            ["Price"],
            ["Comfort", "Max speed"],
            ["Comfort", "Price", "Rating"],
            [],
            ["Max speed", "Price"],
            [],
        ],
    ]

    cv_matrix: pd.DataFrame = counter_veto(performance_table, scales, counter_veto_thresholds)

    assert cv_matrix.index.equals(performance_table.index)
    assert cv_matrix.columns.equals(performance_table.index)

    helpers.assert_cv_criteria_names(expected_values, cv_matrix.to_numpy())


def test_counter_veto_profiles(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    counter_veto_thresholds: pd.Series,
    profiles_performance: pd.DataFrame,
) -> None:
    expected_alternatives_profiles = [
        [
            [],
            [],
            [],
            ["Latency time"],
            ["Latency time"],
        ],
        [
            [],
            ["Comfort"],
            ["Comfort", "Repair cost"],
            ["Comfort", "Rating", "Repair cost"],
            ["Comfort", "Latency time", "Max speed", "Rating", "Repair cost"],
        ],
        [
            [],
            ["Latency time"],
            ["Latency time"],
            ["Latency time", "Rating"],
            ["Latency time", "Price", "Rating"],
        ],
        [
            [],
            ["Max speed"],
            ["Max speed", "Price"],
            ["Max speed", "Price"],
            ["Max speed", "Price", "Repair cost"],
        ],
        [
            [],
            [],
            ["Max speed", "Price"],
            ["Max speed", "Price"],
            ["Comfort", "Max speed", "Price"],
        ],
        [
            [],
            ["Comfort", "Price"],
            ["Comfort", "Price"],
            ["Comfort", "Max speed", "Price"],
            ["Comfort", "Latency time", "Max speed", "Price"],
        ],
        [
            [],
            ["Latency time"],
            ["Latency time", "Repair cost"],
            ["Latency time", "Repair cost"],
            ["Latency time", "Repair cost"],
        ],
        [
            [],
            ["Max speed", "Price"],
            ["Max speed", "Price"],
            ["Max speed", "Price", "Rating"],
            ["Max speed", "Price", "Rating"],
        ],
        [
            [],
            ["Price"],
            ["Price"],
            ["Price", "Rating"],
            ["Price", "Rating", "Repair cost"],
        ],
        [
            [],
            ["Price"],
            ["Price"],
            ["Max speed", "Price"],
            ["Max speed", "Price"],
        ],
        [
            [],
            ["Price"],
            ["Max speed", "Price"],
            ["Comfort", "Max speed", "Price"],
            ["Comfort", "Max speed", "Price"],
        ],
        [
            [],
            [],
            [],
            [],
            ["Latency time", "Price", "Rating"],
        ],
        [
            [],
            ["Max speed", "Price"],
            ["Comfort", "Max speed", "Price"],
            ["Comfort", "Max speed", "Price"],
            ["Comfort", "Max speed", "Price"],
        ],
    ]
    expected_profiles_alternatives = [
        [
            ["Comfort", "Max speed", "Price", "Rating", "Repair cost"],
            ["Price"],
            ["Max speed", "Price", "Repair cost"],
            ["Latency time", "Price", "Rating"],
            ["Latency time", "Price", "Repair cost"],
            ["Price", "Rating", "Repair cost"],
            ["Comfort", "Price", "Rating"],
            ["Latency time", "Price", "Repair cost"],
            ["Comfort", "Latency time", "Max speed"],
            ["Comfort", "Latency time", "Price", "Rating", "Repair cost"],
            ["Latency time", "Rating", "Repair cost"],
            ["Max speed", "Price", "Repair cost"],
            ["Latency time", "Repair cost"],
        ],
        [
            ["Rating", "Repair cost"],
            ["Price"],
            ["Repair cost"],
            ["Rating"],
            ["Repair cost"],
            ["Rating", "Repair cost"],
            ["Price", "Rating"],
            ["Latency time", "Repair cost"],
            [],
            ["Rating", "Repair cost"],
            ["Latency time", "Rating", "Repair cost"],
            ["Repair cost"],
            ["Latency time", "Repair cost"],
        ],
        [
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            ["Latency time"],
            [],
            [],
            ["Latency time"],
            [],
            ["Latency time"],
        ],
        [
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            ["Latency time"],
            [],
            [],
            ["Latency time"],
            [],
            [],
        ],
        [
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        ],
    ]

    cv_matrix_alt_prof, cv_matrix_prof_alt = counter_veto(
        performance_table, scales, counter_veto_thresholds, profiles_performance
    )

    assert cv_matrix_alt_prof.index.equals(performance_table.index)
    assert cv_matrix_alt_prof.columns.equals(profiles_performance.index)

    assert cv_matrix_prof_alt.index.equals(profiles_performance.index)
    assert cv_matrix_prof_alt.columns.equals(performance_table.index)

    helpers.assert_cv_criteria_names(expected_alternatives_profiles, cv_matrix_alt_prof.to_numpy())
    helpers.assert_cv_criteria_names(expected_profiles_alternatives, cv_matrix_prof_alt.to_numpy())


def test_counter_veto_count_no_profiles(
    performance_table: pd.DataFrame, scales: pd.Series, counter_veto_thresholds: pd.Series
) -> None:
    expected_values = [
        [0, 0, 1, 1, 1, 1, 0, 2, 1, 1, 1, 1, 1],
        [3, 0, 1, 1, 2, 2, 2, 2, 1, 2, 3, 2, 2],
        [1, 1, 0, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1],
        [1, 1, 2, 0, 0, 1, 2, 2, 2, 0, 2, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 0, 2, 0, 1, 1, 0],
        [3, 1, 2, 1, 1, 0, 2, 1, 2, 1, 1, 2, 0],
        [0, 1, 1, 1, 2, 2, 0, 2, 1, 1, 2, 2, 2],
        [3, 1, 2, 2, 1, 1, 3, 0, 2, 1, 1, 2, 0],
        [2, 1, 2, 2, 2, 2, 2, 2, 0, 1, 3, 2, 0],
        [2, 1, 3, 1, 1, 1, 1, 1, 1, 0, 2, 2, 0],
        [3, 1, 2, 1, 1, 0, 1, 0, 2, 0, 0, 2, 0],
        [0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 2, 0, 0],
        [3, 1, 2, 1, 1, 1, 2, 1, 2, 3, 0, 2, 0],
    ]

    cv_counted_matrix: pd.DataFrame = counter_veto_count(
        performance_table, scales, counter_veto_thresholds
    )

    assert cv_counted_matrix.index.equals(performance_table.index)
    assert cv_counted_matrix.columns.equals(performance_table.index)

    helpers.assert_array_values(expected_values, cv_counted_matrix.to_numpy())


def test_counter_veto_count_profiles(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    counter_veto_thresholds: pd.Series,
    profiles_performance: pd.DataFrame,
) -> None:
    expected_alternatives_profiles = [
        [0, 0, 0, 1, 1],
        [0, 1, 2, 3, 5],
        [0, 1, 1, 2, 3],
        [0, 1, 2, 2, 3],
        [0, 0, 2, 2, 3],
        [0, 2, 2, 3, 4],
        [0, 1, 2, 2, 2],
        [0, 2, 2, 3, 3],
        [0, 1, 1, 2, 3],
        [0, 1, 1, 2, 2],
        [0, 1, 2, 3, 3],
        [0, 0, 0, 0, 3],
        [0, 2, 3, 3, 3],
    ]
    expected_profiles_alternatives = [
        [5, 1, 3, 3, 3, 3, 3, 3, 3, 5, 3, 3, 2],
        [2, 1, 1, 1, 1, 2, 2, 2, 0, 2, 3, 1, 2],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    cv_matrix_alt_prof, cv_matrix_prof_alt = counter_veto_count(
        performance_table, scales, counter_veto_thresholds, profiles_performance
    )

    assert cv_matrix_alt_prof.index.equals(performance_table.index)
    assert cv_matrix_alt_prof.columns.equals(profiles_performance.index)

    assert cv_matrix_prof_alt.index.equals(profiles_performance.index)
    assert cv_matrix_prof_alt.columns.equals(performance_table.index)

    helpers.assert_array_values(expected_alternatives_profiles, cv_matrix_alt_prof.to_numpy())
    helpers.assert_array_values(expected_profiles_alternatives, cv_matrix_prof_alt.to_numpy())
