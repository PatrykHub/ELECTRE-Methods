from typing import Tuple

import pandas as pd
import pytest

from mcda.electre.outranking.sorting import (
    assign_tri_b_class,
    assign_tri_c_class,
    assign_tri_nc_class,
    assign_tri_nb_class,
    assign_tri_rc_class,
)


@pytest.fixture
def boundary_profiles_tri_b() -> pd.Series:
    return pd.Series(["p1", "p2", None], index=["Bad", "Medium", "Good"])


@pytest.fixture
def crisp_outranking() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(
            [
                [True, True],
                [True, False],
                [True, False],
                [False, False],
                [True, False],
                [True, False],
            ],
            index=[
                "Audi A3",
                "Audi A4",
                "BMW 118",
                "BMW 320",
                "Volvo C30",
                "Volvo S40",
            ],
            columns=["p1", "p2"],
        ),
        pd.DataFrame(
            [
                [False, False, False, False, False, False],
                [True, True, True, True, True, True],
            ],
            index=["p1", "p2"],
            columns=[
                "Audi A3",
                "Audi A4",
                "BMW 118",
                "BMW 320",
                "Volvo C30",
                "Volvo S40",
            ],
        ),
    )


@pytest.fixture
def expected() -> pd.Series:
    return pd.Series(
        [
            ("Good", "Good"),
            ("Medium", "Medium"),
            ("Medium", "Medium"),
            ("Bad", "Medium"),
            ("Medium", "Medium"),
            ("Medium", "Medium"),
        ],
        index=["Audi A3", "Audi A4", "BMW 118", "BMW 320", "Volvo C30", "Volvo S40"],
    )


def test_assign_tri_b_class(
        crisp_outranking, boundary_profiles_tri_b, expected
) -> None:
    assert assign_tri_b_class(
        crisp_outranking[0],
        crisp_outranking[1],
        boundary_profiles_tri_b,
    ).equals(expected)


@pytest.fixture
def characteristic_profiles_tri_c() -> pd.Series:
    return pd.Series(
        ["C1", "C2", "C3", "C4", "C5"], index=["Bad", "Poor", "Okay", "Good", "Perfect"]
    )


@pytest.fixture
def outranking_tri_c() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(
            [
                [True, True, True, False, False],
                [True, True, True, False, False],
                [True, True, True, False, False],
                [True, True, True, False, False],
                [True, True, False, False, False],
                [True, True, False, False, False],
                [True, True, False, False, False],
                [True, True, True, True, False],
                [True, True, False, False, False],
                [True, True, False, False, False],
                [True, True, False, False, False],
            ],
            index=[
                "France",
                "Italy",
                "Spain",
                "Germany",
                "Sweden",
                "Denmark",
                "Russia",
                "Luxembourg",
                "Portugal",
                "Greece",
                "Poland",
            ],
            columns=["C1", "C2", "C3", "C4", "C5"],
        ),
        pd.DataFrame(
            [
                [False, False, False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, False, False, False],
                [False, False, False, True, False, False, False, True, False, False, False],
                [True, True, True, True, True, True, True, True, False, False, False],
                [True, True, True, True, True, True, True, True, True, True, True],
            ],
            index=["C1", "C2", "C3", "C4", "C5"],
            columns=[
                "France",
                "Italy",
                "Spain",
                "Germany",
                "Sweden",
                "Denmark",
                "Russia",
                "Luxembourg",
                "Portugal",
                "Greece",
                "Poland",
            ],
        ),
    )


@pytest.fixture
def credibility_tri_c() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(
            [
                [1.0, 1.0, 0.9, 0.5, 0.3],
                [1.0, 1.0, 0.9, 0.6, 0.3],
                [1.0, 1.0, 0.9, 0.6, 0.3],
                [1.0, 0.9, 0.7, 0.3, 0.1],
                [1.0, 0.9, 0.6, 0.3, 0.1],
                [1.0, 0.9, 0.6, 0.5, 0.1],
                [1.0, 0.9, 0.4, 0.3, 0.1],
                [1.0, 1.0, 1.0, 0.8, 0.3],
                [1.0, 0.9, 0.6, 0.5, 0.3],
                [1.0, 0.9, 0.6, 0.6, 0.5],
                [1.0, 0.9, 0.4, 0.3, 0.2],
            ],
            index=[
                "France",
                "Italy",
                "Spain",
                "Germany",
                "Sweden",
                "Denmark",
                "Russia",
                "Luxembourg",
                "Portugal",
                "Greece",
                "Poland",
            ],
            columns=["C1", "C2", "C3", "C4", "C5"],
        ),
        pd.DataFrame(
            [
                [0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.3, 0.3, 0.5, 0.5, 0.2, 0.2, 0.2, 0.5],
                [0.6, 0.5, 0.6, 0.7, 0.5, 0.4, 0.6, 0.9, 0.3, 0.3, 0.5],
                [0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 1.0, 0.5, 0.4, 0.6],
                [0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            index=["C1", "C2", "C3", "C4", "C5"],
            columns=[
                "France",
                "Italy",
                "Spain",
                "Germany",
                "Sweden",
                "Denmark",
                "Russia",
                "Luxembourg",
                "Portugal",
                "Greece",
                "Poland",
            ],
        ),
    )


@pytest.fixture
def expected_tri_c() -> pd.Series:
    return pd.Series(
        [
            ("Okay", "Okay"),
            ("Good", "Good"),
            ("Okay", "Good"),
            ("Okay", "Okay"),
            ("Okay", "Okay"),
            ("Okay", "Good"),
            ("Poor", "Okay"),
            ("Okay", "Good"),
            ("Okay", "Good"),
            ("Okay", "Perfect"),
            ("Poor", "Good"),
        ],
        index=[
            "France",
            "Italy",
            "Spain",
            "Germany",
            "Sweden",
            "Denmark",
            "Russia",
            "Luxembourg",
            "Portugal",
            "Greece",
            "Poland",
        ],
    )

@pytest.fixture
def categories_tri_c_2() -> pd.Series:
    return pd.Series(
        ["C0", "C1", "C2", "C3", "C4", "C5", "C6"], index=["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    )


@pytest.fixture
def crisp_outranking_ap_tri_c_2() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [True, True, False, False, False, False, False],
            [True, True, True, True, True, True, False],
            [True, True, True, True, False, False, False],
            [True, True, True, False, False, False, False],
            [True, True, False, False, False, False, False],
            [True, True, False, False, False, False, False],
            [True, True, True, False, False, False, False],
            [True, True, True, True, True, False, False],
            [True, True, False, False, False, False, False],
            [True, True, True, False, False, False, False],
            [True, True, False, False, False, False, False],
            [True, True, True, True, True, True, False],
            [True, True, False, False, False, False, False],
            [True, True, True, False, False, False, False],
            [True, True, True, True, False, False, False],
        ],
        index=[
            "Action1",
            "Action2",
            "Action3",
            "Action4",
            "Action5",
            "Action6",
            "Action7",
            "Action8",
            "Action9",
            "Action10",
            "Action11",
            "Action12",
            "Action13",
            "Action14",
            "Action15",
        ],
        columns=["C0", "C1", "C2", "C3", "C4", "C5", "C6"],
    )


@pytest.fixture
def crisp_outranking_pa_tri_c_2() -> pd.DataFrame:
    return pd.DataFrame(
        [[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
         [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
         [False, False, False, False, True, True, False, False, True, False, True, False, True, False, False],
         [True, False, False, True, True, True, False, False, True, True, True, False, True, False, False],
         [True, False, True, True, True, True, True, False, True, True, True, False, True, True, True],
         [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
         [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
         ],
        index=["C0", "C1", "C2", "C3", "C4", "C5", "C6"],
        columns=[
            "Action1",
            "Action2",
            "Action3",
            "Action4",
            "Action5",
            "Action6",
            "Action7",
            "Action8",
            "Action9",
            "Action10",
            "Action11",
            "Action12",
            "Action13",
            "Action14",
            "Action15",
        ]
    )


@pytest.fixture
def credibility_ap_tri_c_2() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1.0, 0.9, 0.65, 0.45, 0.15, 0.15, 0.0],
            [1.0, 0.9, 0.8, 0.8, 0.7, 0.7, 0.0],
            [1.0, 1.0, 0.9, 0.9, 0.3, 0.15, 0.0],
            [1.0, 0.9, 0.9, 0.65, 0.2, 0.2, 0.0],
            [1.0, 1.0, 0.45, 0.2, 0.1, 0.0, 0.0],
            [1.0, 0.9, 0.55, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.8, 0.65, 0.55, 0.2, 0.0],
            [1.0, 0.9, 0.9, 0.8, 0.8, 0.4, 0.0],
            [1.0, 1.0, 0.55, 0.1, 0.1, 0.0, 0.0],
            [1.0, 0.9, 0.8, 0.35, 0.15, 0.0, 0.0],
            [1.0, 0.75, 0.65, 0.25, 0.15, 0.0, 0.0],
            [1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.0],
            [1.0, 0.9, 0.65, 0.15, 0.15, 0.0, 0.0],
            [1.0, 0.9, 0.9, 0.45, 0.35, 0.0, 0.0],
            [1.0, 1.0, 0.9, 0.8, 0.45, 0.1, 0.0]
        ],
        index=[
            "Action1",
            "Action2",
            "Action3",
            "Action4",
            "Action5",
            "Action6",
            "Action7",
            "Action8",
            "Action9",
            "Action10",
            "Action11",
            "Action12",
            "Action13",
            "Action14",
            "Action15",
        ],
        columns=["C0", "C1", "C2", "C3", "C4", "C5", "C6"],
    )


@pytest.fixture
def credibility_pa_tri_c_2() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.35, 0.2, 0.1, 0.1, 0.55, 0.45, 0.2, 0.1, 0.5, 0.2, 0.45, 0.2, 0.35, 0.2, 0.1],
            [0.55, 0.2, 0.2, 0.35, 0.8, 1.0, 0.35, 0.2, 0.9, 0.65, 0.75, 0.2, 0.85, 0.55, 0.2],
            [0.85, 0.3, 0.6, 0.8, 0.9, 1.0, 0.45, 0.2, 0.9, 0.85, 0.85, 0.2, 0.85, 0.65, 0.55],
            [0.85, 0.3, 0.85, 0.8, 0.9, 1.0, 0.8, 0.65, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 0.9],
            [1.0, 0.85, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        index=["C0", "C1", "C2", "C3", "C4", "C5", "C6"],
        columns=[
            "Action1",
            "Action2",
            "Action3",
            "Action4",
            "Action5",
            "Action6",
            "Action7",
            "Action8",
            "Action9",
            "Action10",
            "Action11",
            "Action12",
            "Action13",
            "Action14",
            "Action15",
        ]
    )


@pytest.fixture
def expected_tri_c_2() -> pd.Series:
    return pd.Series(
        [
            ("C2", "C2"),
            ("C5", "C5"),
            ("C3", "C3"),
            ("C3", "C3"),
            ("C1", "C1"),
            ("C2", "C2"),
            ("C3", "C4"),
            ("C4", "C4"),
            ("C2", "C2"),
            ("C2", "C2"),
            ("C2", "C2"),
            ("C5", "C5"),
            ("C2", "C2"),
            ("C2", "C3"),
            ("C3", "C3")
        ],
        index=[
            "Action1",
            "Action2",
            "Action3",
            "Action4",
            "Action5",
            "Action6",
            "Action7",
            "Action8",
            "Action9",
            "Action10",
            "Action11",
            "Action12",
            "Action13",
            "Action14",
            "Action15",
        ])


def test_assign_tri_c_class_2(
        crisp_outranking_ap_tri_c_2: pd.DataFrame,
        crisp_outranking_pa_tri_c_2: pd.DataFrame,
        credibility_ap_tri_c_2: pd.DataFrame,
        credibility_pa_tri_c_2: pd.DataFrame,
        categories_tri_c_2: pd.Series,
        expected_tri_c_2: pd.Series,
) -> None:
    assert assign_tri_c_class(
        crisp_outranking_ap_tri_c_2, crisp_outranking_pa_tri_c_2, credibility_ap_tri_c_2,
        credibility_pa_tri_c_2, categories_tri_c_2
    ).equals(expected_tri_c_2)


def test_assign_tri_c_class(
        outranking_tri_c,
        credibility_tri_c,
        characteristic_profiles_tri_c,
        expected_tri_c,
) -> None:
    assert assign_tri_c_class(
        outranking_tri_c[0],
        outranking_tri_c[1],
        credibility_tri_c[0],
        credibility_tri_c[1],
        characteristic_profiles_tri_c,
    ).equals(expected_tri_c)


@pytest.fixture
def expected_tri_rc() -> pd.Series:
    pd.Series(
        ["C1", "C2", "C3", "C4", "C5"], index=["Bad", "Poor", "Okay", "Good", "Perfect"]
    )
    return pd.Series(
        [
            ("Okay", "Okay"),
            ("Good", "Good"),
            ("Okay", "Good"),
            ("Okay", "Okay"),
            ("Okay", "Okay"),
            ("Okay", "Good"),
            ("Poor", "Okay"),
            ("Okay", "Good"),
            ("Okay", "Good"),
            ("Okay", "Perfect"),
            ("Poor", "Good"),
        ],
        index=[
            "France",
            "Italy",
            "Spain",
            "Germany",
            "Sweden",
            "Denmark",
            "Russia",
            "Luxembourg",
            "Portugal",
            "Greece",
            "Poland",
        ],
    )


def test_assign_tri_rc_class(
        outranking_tri_c,
        credibility_tri_c,
        expected_tri_rc,
        characteristic_profiles_tri_c,
) -> None:
    assert assign_tri_rc_class(
        outranking_tri_c[0],
        outranking_tri_c[1],
        credibility_tri_c[0],
        credibility_tri_c[1],
        characteristic_profiles_tri_c,
    ).equals(expected_tri_rc)


@pytest.fixture
def categories_tri_nc() -> pd.Series:
    return pd.Series(
        [["C0"], ["C1", "C2"], ["C3", "C4"],
         ["C5"], ["C6"]], index=["C1", "C2", "C3", "C4", "C5"]
    )


@pytest.fixture
def credibility_ap_tri_nc() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [1.0, 0.9, 0.75, 0.65, 0.15, 0.15, 0.1],
            [1.0, 0.9, 0.8, 0.8, 0.7, 0.7, 0.0],
            [1.0, 1.0, 0.7, 0.5, 0.3, 0.15, 0.0],
            [1.0, 0.9, 0.9, 0.45, 0.2, 0.2, 0.0],
            [1.0, 1.0, 0.75, 0.2, 0.1, 0.0, 0.0],
            [1.0, 0.9, 0.85, 0.5, 0.3, 0.0, 0.0],
            [1.0, 1.0, 0.8, 0.65, 0.55, 0.2, 0.0],
            [1.0, 0.9, 0.9, 0.8, 0.8, 0.4, 0.0],
            [1.0, 1.0, 0.55, 0.1, 0.1, 0.0, 0.0],
            [1.0, 0.9, 0.8, 0.35, 0.15, 0.0, 0.0],
            [1.0, 0.85, 0.65, 0.25, 0.15, 0.0, 0.0],
            [1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.0],
            [1.0, 0.9, 0.65, 0.15, 0.15, 0.1, 0.0],
            [1.0, 0.9, 0.9, 0.45, 0.35, 0.3, 0.1],
            [1.0, 1.0, 0.9, 0.8, 0.45, 0.1, 0.0]
        ],
        index=[
            "Action1",
            "Action2",
            "Action3",
            "Action4",
            "Action5",
            "Action6",
            "Action7",
            "Action8",
            "Action9",
            "Action10",
            "Action11",
            "Action12",
            "Action13",
            "Action14",
            "Action15",
        ],
        columns=["C0", "C1", "C2", "C3", "C4", "C5", "C6"],
    )


@pytest.fixture
def credibility_pa_tri_nc() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.35, 0.2, 0.1, 0.1, 0.55, 0.25, 0.2, 0.1, 0.5, 0.2, 0.45, 0.2, 0.85, 0.2, 0.7],
            [0.55, 0.2, 0.2, 0.35, 0.8, 1.0, 0.35, 0.2, 0.9, 0.65, 0.75, 0.2, 0.85, 0.55, 0.2],
            [0.85, 0.3, 1.0, 0.8, 0.9, 1.0, 0.45, 0.2, 0.9, 0.85, 0.85, 0.2, 0.85, 0.65, 0.85],
            [0.85, 0.3, 0.85, 0.8, 0.9, 1.0, 0.8, 0.65, 1.0, 1.0, 1.0, 0.2, 1.0, 1.0, 0.9],
            [1.0, 0.85, 1.0, 1.0, 1.0, 0.85, 1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        index=["C0", "C1", "C2", "C3", "C4", "C5", "C6"],
        columns=[
            "Action1",
            "Action2",
            "Action3",
            "Action4",
            "Action5",
            "Action6",
            "Action7",
            "Action8",
            "Action9",
            "Action10",
            "Action11",
            "Action12",
            "Action13",
            "Action14",
            "Action15",
        ]
    )


@pytest.fixture
def expected_tri_nc() -> pd.Series:
    return pd.Series(
        [
            ("C3", "C3"),
            ("C4", "C4"),
            ("C3", "C3"),
            ("C3", "C3"),
            ("C2", "C2"),
            ("C2", "C2"),
            ("C3", "C3"),
            ("C3", "C3"),
            ("C2", "C2"),
            ("C2", "C2"),
            ("C1", "C2"),
            ("C4", "C4"),
            ("C2", "C2"),
            ("C2", "C2"),
            ("C2", "C3")
        ],
        index=[
            "Action1",
            "Action2",
            "Action3",
            "Action4",
            "Action5",
            "Action6",
            "Action7",
            "Action8",
            "Action9",
            "Action10",
            "Action11",
            "Action12",
            "Action13",
            "Action14",
            "Action15",
        ])


def test_assign_tri_nc_class(
        credibility_ap_tri_nc: pd.DataFrame,
        credibility_pa_tri_nc: pd.DataFrame,
        categories_tri_nc: pd.Series,
        expected_tri_nc: pd.Series,
) -> None:
    assert assign_tri_nc_class(
        credibility_ap_tri_nc, credibility_pa_tri_nc, categories_tri_nc
    ).equals(expected_tri_nc)


@pytest.mark.parametrize(
    (
        "crisp_outranking_ap",
        "crisp_outranking_pa",
        "categories",
        "expected",
    ),
    (
        (
            pd.DataFrame(
                [
                    [0, 1, 1, 1],
                    [0, 0, 0, 0],
                    [0, 0, 1, 1],
                    [0, 1, 0, 0],
                    [1, 1, 1, 0],
                ],
                index=["Audi", "BMW", "Fiat", "Honda", "Opel"],
                columns=["p1_1", "p1_2", "p2_1", "p2_2"],
            ),
            pd.DataFrame(
                [[0, 1, 1, 1, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 1, 0, 1]],
                index=["p1_1", "p1_2", "p2_1", "p2_2"],
                columns=["Audi", "BMW", "Fiat", "Honda", "Opel"],
            ),
            pd.Series(
                [["p1_1", "p1_2"], ["p2_1", "p2_2"], []],
                index=["good", "medium", "bad"],
            ),
            pd.Series(
                [
                    ("good", "good"),
                    ("bad", "bad"),
                    ("medium", "medium"),
                    ("bad", "good"),
                    ("good", "good"),
                ],
                index=["Audi", "BMW", "Fiat", "Honda", "Opel"],
            ),
        ),
    ),
)
def test_assign_tri_nb_class(
    crisp_outranking_ap: pd.DataFrame,
    crisp_outranking_pa: pd.DataFrame,
    categories: pd.Series,
    expected: pd.Series,
) -> None:
    assert assign_tri_nb_class(
        crisp_outranking_ap, crisp_outranking_pa, categories
    ).equals(expected)
