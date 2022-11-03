from typing import List

import numpy as np
import pandas as pd
import pytest

from mcda.core.functions import Threshold
from mcda.electre.concordance import concordance, concordance_reinforced

from .. import helpers


@pytest.fixture
def reinforcement_thresholds(
    criterion_names: List[str],
) -> pd.Series:
    """Returns all criteria reinforcement thresholds."""
    return pd.Series(
        [
            Threshold(0.3, 30000),
            Threshold(0.15, 10000),
            None,
            Threshold(0, 90),
            Threshold(0, 27),
            None,
            None,
            Threshold(0, 70),
        ],
        index=criterion_names,
    )


@pytest.fixture
def reinforcement_factors(
    criterion_names: List[str],
) -> pd.Series:
    """Returns all criteria reinforcement factors."""
    return pd.Series(
        [1.3, 1.2, None, 1.01, 1.1, None, None, 1.5], index=criterion_names
    )


@pytest.fixture
def interactions(criterion_names: List[str]) -> pd.DataFrame:
    """Returns matrix with all interactions between criteria."""
    return pd.DataFrame([], index=criterion_names, columns=criterion_names)


def test_concordance_no_profiles(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    weights: pd.Series,
    indifference_thresholds: pd.Series,
    preference_thresholds: pd.Series,
) -> None:
    expected_values: List[List[float]] = [
        [
            1.0,
            0.336842105263,
            0.641403508772,
            0.574754614737,
            0.473873751579,
            0.578947368421,
            0.456280701754,
            0.474105263158,
            0.715789473684,
            0.747368421053,
            0.578947368421,
            0.747368421053,
            0.494736842105,
        ],
        [
            0.681647431579,
            1.0,
            0.663157894737,
            0.747368421053,
            0.76,
            0.873684210526,
            0.789473684211,
            0.681684210526,
            0.8,
            0.848421052632,
            0.612631578947,
            0.873684210526,
            0.747368421053,
        ],
        [
            0.789473684211,
            0.461894736842,
            1.0,
            0.646982488421,
            0.619196362105,
            0.578947368421,
            0.498947368421,
            0.747368421053,
            0.627789473684,
            0.536842105263,
            0.651705263158,
            0.789473684211,
            0.368421052632,
        ],
        [
            0.789473684211,
            0.326315789474,
            0.663157894737,
            1.0,
            0.578947368421,
            0.507368421053,
            0.578947368421,
            0.452631578947,
            0.615157894737,
            0.705263157895,
            0.536842105263,
            0.724912280702,
            0.621052631579,
        ],
        [
            0.578947368421,
            0.336842105263,
            0.631578947368,
            0.715789473684,
            1.0,
            0.500350877193,
            0.578947368421,
            0.621052631579,
            0.505263157895,
            0.663157894737,
            0.705263157895,
            0.886315789474,
            0.789473684211,
        ],
        [
            0.442947368421,
            0.631578947368,
            0.631578947368,
            0.571789473684,
            0.631578947368,
            1.0,
            0.421052631579,
            0.8,
            0.578811808421,
            0.789473684211,
            0.765779113684,
            0.631578947368,
            0.490526315789,
        ],
        [
            0.873684210526,
            0.673684210526,
            0.547368421053,
            0.656505263158,
            0.451368421053,
            0.633684210526,
            1.0,
            0.494736842105,
            0.715789473684,
            0.776842105263,
            0.578947368421,
            0.789473684211,
            0.494736842105,
        ],
        [
            0.578947368421,
            0.421052631579,
            0.750315789474,
            0.673684210526,
            0.747368421053,
            0.789473684211,
            0.578947368421,
            1.0,
            0.589473684211,
            0.747368421053,
            0.815630525716,
            0.747368421053,
            0.632296488421,
        ],
        [
            0.621052631579,
            0.269894736842,
            0.621052631579,
            0.705263157895,
            0.705263157895,
            0.510175438596,
            0.284210526316,
            0.536842105263,
            1.0,
            0.705263157895,
            0.536842105263,
            0.811929824561,
            0.669369162105,
        ],
        [
            0.647578947368,
            0.252631578947,
            0.463157894737,
            0.633263157895,
            0.673684210526,
            0.558596491228,
            0.284294736842,
            0.488421052632,
            0.797480471579,
            1.0,
            0.531816197895,
            0.481403508772,
            0.557894736842,
        ],
        [
            0.630736842105,
            0.421052631579,
            0.8,
            0.616421052632,
            0.8,
            0.789473684211,
            0.589473684211,
            0.842105263158,
            0.589473684211,
            0.747368421053,
            1.0,
            0.8,
            0.709251331163,
        ],
        [
            0.578947368421,
            0.336842105263,
            0.463157894737,
            0.589473684211,
            0.702315789474,
            0.457684210526,
            0.452631578947,
            0.348210526316,
            0.505263157895,
            0.536842105263,
            0.410526315789,
            1.0,
            0.578947368421,
        ],
        [
            0.505263157895,
            0.252631578947,
            0.631578947368,
            0.689122807018,
            0.738245614035,
            0.715789473684,
            0.505263157895,
            0.673684210526,
            0.433684210526,
            0.626105263158,
            0.757894736842,
            0.698105263158,
            1.0,
        ],
    ]

    concordance_matrix: pd.DataFrame = concordance(
        performance_table,
        scales,
        weights,
        indifference_thresholds,
        preference_thresholds,
    )

    assert concordance_matrix.index.equals(performance_table.index)
    assert concordance_matrix.columns.equals(performance_table.index)

    helpers.assert_array_values(expected_values, concordance_matrix.to_numpy())


def test_concordance_profiles(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    weights: pd.Series,
    indifference_thresholds: pd.Series,
    preference_thresholds: pd.Series,
    profiles_performance: pd.DataFrame,
) -> None:
    expected_alternatives_profiles: List[List[float]] = [
        [0.210526315789, 0.460163330526, 0.732631578947, 1.0, 1.0],
        [0.547368421053, 0.663157894737, 1.0, 1.0, 1.0],
        [0.368421052632, 0.74291752, 0.789473684211, 0.789473684211, 0.953135482238],
        [0.2, 0.494736842105, 0.816421052632, 1.0, 1.0],
        [0.294736842105, 0.589473684211, 0.810105263158, 1.0, 1.0],
        [0.463157894737, 0.618947368421, 0.621754385965, 0.744, 0.883731848018],
        [0.273684210526, 0.505263157895, 0.807376018947, 1.0, 1.0],
        [
            0.340412909474,
            0.673684210526,
            0.673684210526,
            0.747368421053,
            0.855631836236,
        ],
        [
            0.392896109474,
            0.477894736842,
            0.755789473684,
            0.831578947368,
            0.831578947368,
        ],
        [0.294736842105, 0.589473684211, 0.8, 0.874947368421, 0.926315789474],
        [
            0.391236061053,
            0.589473684211,
            0.747298116041,
            0.931789473684,
            0.978947368421,
        ],
        [0.168421052632, 0.590877192982, 0.909473684211, 1.0, 1.0],
        [
            0.421052631579,
            0.454736842105,
            0.884210526316,
            0.888421052632,
            0.926315789474,
        ],
    ]
    expected_profiles_alternatives: List[List[float]] = [
        [
            0.831578947368,
            0.765894736842,
            0.789473684211,
            0.924210526316,
            1.0,
            0.831578947368,
            0.789473684211,
            0.781052631579,
            0.990736842105,
            1.0,
            0.831578947368,
            1.0,
            0.903157894737,
        ],
        [
            0.789473684211,
            0.578947368421,
            0.464,
            0.631578947368,
            0.703578947368,
            0.410526315789,
            0.789473684211,
            0.410526315789,
            0.715789473684,
            0.747368421053,
            0.410526315789,
            0.819115789474,
            0.578947368421,
        ],
        [
            0.578947368421,
            0.244210526316,
            0.463157894737,
            0.294736842105,
            0.410526315789,
            0.410526315789,
            0.441852631579,
            0.326315789474,
            0.353684210526,
            0.536842105263,
            0.410526315789,
            0.584551633684,
            0.578947368421,
        ],
        [
            0.460646178947,
            0.126315789474,
            0.373375442105,
            0.294736842105,
            0.252631578947,
            0.392280701754,
            0.190315789474,
            0.252631578947,
            0.336842105263,
            0.368421052632,
            0.410526315789,
            0.367719298246,
            0.494736842105,
        ],
        [
            0.340210526316,
            0.126315789474,
            0.336842105263,
            0.199578947368,
            0.210526315789,
            0.294736842105,
            0.126315789474,
            0.252631578947,
            0.294736842105,
            0.326315789474,
            0.336842105263,
            0.336842105263,
            0.483350202354,
        ],
    ]

    concordance_matrix_alt_prof, concordance_matrix_prof_alt = concordance(
        performance_table,
        scales,
        weights,
        indifference_thresholds,
        preference_thresholds,
        profiles_performance,
    )

    assert concordance_matrix_alt_prof.index.equals(performance_table.index)
    assert concordance_matrix_alt_prof.columns.equals(profiles_performance.index)

    assert concordance_matrix_prof_alt.index.equals(profiles_performance.index)
    assert concordance_matrix_prof_alt.columns.equals(performance_table.index)

    helpers.assert_array_values(
        expected_alternatives_profiles, concordance_matrix_alt_prof.to_numpy()
    )
    helpers.assert_array_values(
        expected_profiles_alternatives, concordance_matrix_prof_alt.to_numpy()
    )

@pytest.mark.skip(reason="XD")
def test_reinforcement_no_profiles(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    weights: pd.Series,
    indifference_thresholds: pd.Series,
    preference_thresholds: pd.Series,
    reinforcement_thresholds: pd.Series,
    reinforcement_factors: pd.Series,
) -> None:
    expected_values = [
        [
            1.0,
            0.350515463918,
            0.655892255892,
            0.583522560825,
            0.505128776238,
            0.60396039604,
            0.456280701754,
            0.505346534653,
            0.721649484536,
            0.752577319588,
            0.60396039604,
            0.762376237624,
            0.524752475248,
        ],
        [
            0.699727025417,
            1.0,
            0.681908548708,
            0.766081871345,
            0.774257425743,
            0.878787878788,
            0.792960662526,
            0.705263157895,
            0.815031152648,
            0.859649122807,
            0.635643564356,
            0.880715705765,
            0.762376237624,
        ],
        [
            0.792960662526,
            0.472989690722,
            1.0,
            0.6542612,
            0.627047983505,
            0.587628865979,
            0.498947368421,
            0.752577319588,
            0.641379310345,
            0.553752535497,
            0.658886597938,
            0.79381443299,
            0.381443298969,
        ],
        [
            0.789739276703,
            0.32716568545,
            0.677158999193,
            1.0,
            0.59595959596,
            0.527272727273,
            0.579478553406,
            0.485148514851,
            0.62200165426,
            0.705263157895,
            0.564356435644,
            0.736346516008,
            0.636363636364,
        ],
        [
            0.586435070306,
            0.336842105263,
            0.646892655367,
            0.715789473684,
            1.0,
            0.520538720539,
            0.585921325052,
            0.643564356436,
            0.51406120761,
            0.668737060041,
            0.711340206186,
            0.888337468983,
            0.789473684211,
        ],
        [
            0.472488038278,
            0.645030425963,
            0.651116427432,
            0.578881987578,
            0.631578947368,
            1.0,
            0.451097804391,
            0.80412371134,
            0.586301921009,
            0.792960662526,
            0.770608410309,
            0.651116427432,
            0.501030927835,
        ],
        [
            0.878934624697,
            0.680412371134,
            0.565656565657,
            0.676910891089,
            0.48396039604,
            0.655445544554,
            1.0,
            0.524752475248,
            0.736842105263,
            0.790099009901,
            0.60396039604,
            0.80198019802,
            0.524752475248,
        ],
        [
            0.601275917065,
            0.442190669371,
            0.759724473258,
            0.685598377282,
            0.756592292089,
            0.789473684211,
            0.601275917065,
            1.0,
            0.596774193548,
            0.751552795031,
            0.815630525716,
            0.756888168558,
            0.632296488421,
        ],
        [
            0.634888438134,
            0.296551724138,
            0.649122807018,
            0.716024340771,
            0.727095516569,
            0.529966329966,
            0.310344827586,
            0.564356435644,
            1.0,
            0.705263157895,
            0.564356435644,
            0.825860948668,
            0.682727983838,
        ],
        [
            0.660858995138,
            0.279918864097,
            0.503504672897,
            0.646653144016,
            0.686868686869,
            0.576430976431,
            0.310425963489,
            0.518811881188,
            0.797735962994,
            1.0,
            0.55962909703,
            0.520379023884,
            0.575757575758,
        ],
        [
            0.650318979266,
            0.442190669371,
            0.807536466775,
            0.630425963489,
            0.807302231237,
            0.789473684211,
            0.610778443114,
            0.842105263158,
            0.596774193548,
            0.751552795031,
            1.0,
            0.810606060606,
            0.709251331163,
        ],
        [
            0.578947368421,
            0.336842105263,
            0.484848484848,
            0.589473684211,
            0.702315789474,
            0.479595959596,
            0.452631578947,
            0.386930693069,
            0.513457556936,
            0.536842105263,
            0.422680412371,
            1.0,
            0.587628865979,
        ],
        [
            0.531499202552,
            0.280794165316,
            0.659267912773,
            0.705256154358,
            0.747802569304,
            0.727272727273,
            0.531499202552,
            0.686868686869,
            0.443755169562,
            0.632298136646,
            0.772277227723,
            0.714114832536,
            1.0,
        ],
    ]

    concordance_matrix = concordance_reinforced(
        performance_table,
        scales,
        weights,
        indifference_thresholds,
        preference_thresholds,
        reinforcement_thresholds,
        reinforcement_factors,
    )

    assert concordance_matrix.index.equals(performance_table.index)
    assert concordance_matrix.columns.equals(performance_table.index)
    helpers.assert_array_values(expected_values, concordance_matrix.to_numpy())


@pytest.mark.skip(reason="XD")
def test_reinforcement_profiles(
    performance_table: pd.DataFrame,
    scales: pd.Series,
    weights: pd.Series,
    indifference_thresholds: pd.Series,
    preference_thresholds: pd.Series,
    reinforcement_thresholds: pd.Series,
    reinforcement_factors: pd.Series,
    profiles_performance: pd.DataFrame,
) -> None:
    expected_alternatives_profiles = [
        [0.210526315789, 0.471293983505, 0.748514851485, 1.0, 1.0],
        [0.547368421053, 0.668737060041, 1.0, 1.0, 1.0],
        [0.381443298969, 0.75230389858, 0.797160243408, 0.797160243408, 0.954846559966],
        [0.2, 0.495374264087, 0.82405165456, 1.0, 1.0],
        [0.294736842105, 0.596774193548, 0.813482216708, 1.0, 1.0],
        [0.463157894737, 0.63872255489, 0.641812865497, 0.757575757576, 0.892049702519],
        [0.288659793814, 0.515463917526, 0.818819027723, 1.0, 1.0],
        [
            0.340412909474,
            0.690988835726,
            0.690988835726,
            0.760765550239,
            0.863287723709,
        ],
        [0.392896109474, 0.496957403651, 0.7738791423, 0.844054580897, 0.847036328872],
        [
            0.294736842105,
            0.604462474645,
            0.815031152648,
            0.884345794393,
            0.933155080214,
        ],
        [
            0.391236061053,
            0.611244019139,
            0.760698973523,
            0.935406698565,
            0.980063795853,
        ],
        [0.168421052632, 0.590877192982, 0.909473684211, 1.0, 1.0],
        [0.421052631579, 0.4836523126, 0.890350877193, 0.894338118022, 0.930223285486],
    ]
    expected_profiles_alternatives = [
        [
            0.846625766871,
            0.787380497132,
            0.808282208589,
            0.932203389831,
            1.0,
            0.847036328872,
            0.808061420345,
            0.794059405941,
            0.991433021807,
            1.0,
            0.841584158416,
            1.0,
            0.908910891089,
        ],
        [
            0.79797979798,
            0.59595959596,
            0.485656565657,
            0.653465346535,
            0.721188118812,
            0.434343434343,
            0.789473684211,
            0.445544554455,
            0.736842105263,
            0.762376237624,
            0.445544554455,
            0.826424242424,
            0.60396039604,
        ],
        [
            0.578947368421,
            0.244210526316,
            0.484848484848,
            0.309278350515,
            0.422680412371,
            0.434343434343,
            0.441852631579,
            0.366336633663,
            0.367010309278,
            0.546391752577,
            0.422680412371,
            0.584551633684,
            0.587628865979,
        ],
        [
            0.460646178947,
            0.126315789474,
            0.398693606061,
            0.294736842105,
            0.252631578947,
            0.416835016835,
            0.190315789474,
            0.29702970297,
            0.336842105263,
            0.368421052632,
            0.422680412371,
            0.367719298246,
            0.505154639175,
        ],
        [
            0.340210526316,
            0.126315789474,
            0.336842105263,
            0.199578947368,
            0.210526315789,
            0.323232323232,
            0.126315789474,
            0.282828282828,
            0.294736842105,
            0.326315789474,
            0.336842105263,
            0.336842105263,
            0.483350202354,
        ],
    ]

    concordance_matrix_alt_prof, concordance_matrix_prof_alt = concordance_reinforced(
        performance_table,
        scales,
        weights,
        indifference_thresholds,
        preference_thresholds,
        reinforcement_thresholds,
        reinforcement_factors,
        profiles_performance,
    )

    assert concordance_matrix_alt_prof.index.equals(performance_table.index)
    assert concordance_matrix_alt_prof.columns.equals(profiles_performance.index)

    assert concordance_matrix_prof_alt.index.equals(profiles_performance.index)
    assert concordance_matrix_prof_alt.columns.equals(performance_table.index)

    helpers.assert_array_values(
        expected_alternatives_profiles, concordance_matrix_alt_prof.to_numpy()
    )
    helpers.assert_array_values(
        expected_profiles_alternatives, concordance_matrix_prof_alt.to_numpy()
    )
