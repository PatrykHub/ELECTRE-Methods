from typing import Tuple

import pandas as pd
import pytest

from mcda.electre.outranking import (
    assign_tri_class,
    assign_tri_c_class,
    assign_tri_rc_class,
)


@pytest.fixture
def categories_profiles() -> pd.Series:
    return pd.Series([("Bad", "Medium"), ("Medium", "Good")], index=["p1", "p2"])


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


def test_assign_tri_class(
        alternatives, categories_rank, categories_profiles, crisp_outranking, expected
) -> None:
    assert assign_tri_class(
        categories_profiles,
        crisp_outranking[0],
        crisp_outranking[1],
    ).equals(expected)


@pytest.fixture
def categories_profiles_tri_c() -> pd.Series:
    return pd.Series(
        ["C1", "C2", "C3", "C4", "C5"], index=["b1", "b2", "b3", "b4", "b5"]
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
            ("C3", "C3"),
            ("C4", "C4"),
            ("C3", "C4"),
            ("C3", "C3"),
            ("C3", "C3"),
            ("C3", "C4"),
            ("C2", "C3"),
            ("C3", "C4"),
            ("C3", "C4"),
            ("C3", "C5"),
            ("C2", "C4"),
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


def test_assign_tri_c_class(
        categories_profiles_tri_c,
        outranking_tri_c,
        credibility_tri_c,
        expected_tri_c,
) -> None:
    assert assign_tri_c_class(
        categories_profiles_tri_c,
        outranking_tri_c[0],
        outranking_tri_c[1],
        credibility_tri_c[0],
        credibility_tri_c[1],
    ).equals(expected_tri_c)


@pytest.fixture
def expected_tri_rc() -> pd.Series:
    return pd.Series(
        [
            ("C3", "C3"),
            ("C4", "C4"),
            ("C3", "C4"),
            ("C3", "C3"),
            ("C3", "C3"),
            ("C3", "C4"),
            ("C2", "C3"),
            ("C3", "C4"),
            ("C3", "C4"),
            ("C3", "C5"),
            ("C2", "C4"),
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
        alternatives_tri_c,
        categories_rank_tri_c,
        categories_profiles_tri_c,
        outranking_tri_c,
        credibility_tri_c,
        expected_tri_rc,
) -> None:
    assert assign_tri_rc_class(
        categories_profiles_tri_c,
        outranking_tri_c[0],
        outranking_tri_c[1],
        credibility_tri_c[0],
        credibility_tri_c[1],
    ).equals(expected_tri_rc)
