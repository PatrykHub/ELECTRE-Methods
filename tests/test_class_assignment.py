import pandas as pd

import pytest

from mcda.electre.class_assignment import assign_tri_class


@pytest.fixture
def alternatives() -> pd.Series:
    return pd.Series(
        ["Audi A3", "Audi A4", "BMW 118", "BMW 320", "Volvo C30", "Volvo S40"],
        index=["A1", "A2", "A3", "A4", "A5", "A6"],
    )


@pytest.fixture
def categories_rank() -> pd.Series:
    return pd.Series(["Bad", "Medium", "Good"], index=[3, 2, 1])


@pytest.fixture
def categories_profiles() -> pd.Series:
    return pd.Series([("Bad", "Medium"), ("Medium", "Good")], index=["p1", "p2"])


@pytest.fixture
def crisp_outranking() -> pd.DataFrame:
    return pd.concat(
        [
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
        ],
        axis=1,
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
    assert (
        assign_tri_class(
            alternatives, categories_rank, categories_profiles, crisp_outranking
        ).all()
        == expected.all()
    )
