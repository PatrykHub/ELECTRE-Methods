from enum import Enum, auto
from typing import List, Tuple

from .aliases import NumericValue


class RelationType(Enum):
    """Enumeration of MCDA relation types."""

    PREFERENCE = auto()
    INDIFFERENCE = auto()
    INCOMPARABLE = auto()

    @classmethod
    def has_value(cls, x: "RelationType") -> bool:
        """Check if value is in enumeration.

        :param x:
        :return:
        """
        return x in cls

    @classmethod
    def content_message(cls) -> str:
        """Return list of items and their values.

        :return:
        """
        s = ", ".join(f"{item}: {item.value}" for item in cls)
        return "RelationType only has following values " + s


def rank_values(
    values: List[NumericValue], reverse: bool = False
) -> List[int]:
    """Rank values.

    First value in ranking order has the smallest rank.
    By default, it ranks how big values are (i.e value with rank ``0`` is
    the biggest value).
    To reverse ranking order, put parameter `reverse` to ``True``.

    :param values:
    :param reverse:
    :return: ranks list
    """
    order = sorted(
        range(len(values)), key=lambda i: values[i], reverse=not reverse
    )
    return [order.index(i) for i, _ in enumerate(values)]


def sort_elements_by_values(
    values: List[NumericValue], *elements: List, reverse: bool = False
):
    """Sort elements by their values.

    By default, elements are sorted by decreasing values.
    To reverse ranking order, put parameter `reverse` to ``True``.

    :param elements: different lists of any elements that must be sorted
    :param values:
    :param reverse:

    .. warning:: input elements are sorted directly and are therefore modified
    """
    ranks = rank_values(values, reverse)
    for i in range(len(elements)):
        for j, v in enumerate([a for _, a in sorted(zip(ranks, elements[i]))]):
            elements[i][j] = v


Relation = Tuple[int, int, RelationType]