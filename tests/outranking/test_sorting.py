import pandas as pd
import pytest

from mcda.electre.outranking import assign_tri_nb_class


@pytest.mark.parametrize(
    (
        "crisp_outranking_ap",
        "crisp_outranking_pa",
        "categories",
        "optimistic",
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
            True,
            pd.Series(
                ["good", "bad", "medium", "good", "good"],
                index=["Audi", "BMW", "Fiat", "Honda", "Opel"],
            ),
        ),
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
            False,
            pd.Series(
                ["good", "bad", "medium", "bad", "good"],
                index=["Audi", "BMW", "Fiat", "Honda", "Opel"],
            ),
        ),
    ),
)
def test_assign_tri_nb_class(
    crisp_outranking_ap: pd.DataFrame,
    crisp_outranking_pa: pd.DataFrame,
    categories: pd.Series,
    optimistic: bool,
    expected: pd.Series,
) -> None:
    assert assign_tri_nb_class(
        crisp_outranking_ap, crisp_outranking_pa, categories, optimistic
    ).equals(expected)