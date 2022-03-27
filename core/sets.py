from typing import List, Set

import numpy as np


def mask_to_int(mask: str) -> int:
    """Convert binary mask to integer.

    :param mask: contains only the binary mask read from left to right
    :return: corresponding integer
    """
    return int(mask, 2)


def int_to_mask(number: int) -> str:
    """Convert integer to binary mask.

    :param number:
    :return: corresponding binary mask
    """
    return f"{number:b}"


def set_to_index(set_: Set, ensemble: List) -> int:
    """Return integer representation of subset of an ensemble.

    :param set_: set of items (no doublons, possibly unordered)
    :param ensemble: ordered list of all possible items (no doublons)
    :return: binary mask
    """
    res = 0
    bit = 1
    for element in ensemble:
        if element in set_:
            res = union(res, bit)
        bit <<= 1
    return res


def index_to_set(number: int, ensemble: List = None) -> Set:
    """Return set represented by given integer.

    :param number:
    :param ensemble: ordered list of all possible items (no doublons)
    :return: set of items represented
    """
    ensemble = [*range(size(number))] if ensemble is None else ensemble
    res = set()
    bit = 1
    for element in ensemble:
        if intersection(number, bit) != 0:
            res.add(element)
        bit <<= 1
    return res


def cardinal_range(*args: int) -> List[int]:
    """Return range iterator ordered by cardinality and set content.

    :param args: argument available to built-in :func:`range` function
    :return:
    """
    return sorted(
        range(*args), key=lambda k: [cardinality(k), *sorted(index_to_set(k))]
    )


def intersection(n1: int, n2: int) -> int:
    """Return intersection of both sets as integer.

    :param n1:
    :param n2:
    :return:
    """
    return n1 & n2


def union(n1: int, n2: int) -> int:
    """Return union of both sets as integer.

    :param n1:
    :param n2:
    :return:
    """
    return n1 | n2


def disjoint_union(n1: int, n2: int) -> int:
    """Return disjoint union set as integer.

    If we note `A` (resp. `B`) the set represented by `n1` (resp. `n2`), this
    returns :math:`\\{ A \\cup B \\} \\setminus \\{ A \\cap B \\}`

    :param n1:
    :param n2:
    :return:
    """
    return n1 ^ n2


def difference(n1: int, n2: int) -> int:
    """Return difference set as integer.

    If we note `A` (resp. `B`) the set represented by `n1` (resp. `n2`), this
    returns :math:`A \\setminus B`

    :param n1:
    :param n2:
    :return:
    """
    return n1 & (~n2)


def complement(number: int, set_size: int = None) -> int:
    """Return the set complement.

    If we note `A` the set represented by `number`, this returns :math:`A^C`

    :param number:
    :param set_size:
        whole ensemble size (smaller complete superset of `number` if ommited)
    :return:
    """
    set_size = size(number) if set_size is None else set_size
    return difference(complete_set(set_size), number)


def cardinality(number: int) -> int:
    """Return cardinality of set represented by given integer.

    :param number:
    :return:
    """
    res = 0
    while number > 0:
        res += number & 1
        number >>= 1
    return res


def size(number: int) -> int:
    """Return size of smaller complete superset.

    :param number:
    :return:
    """
    return int(np.ceil(np.log2(number + 1)))


def complete_set(size_: int) -> int:
    """Return highest cardinal set of given size.

    :param size_:
    :return:
    """
    return 2 ** size_ - 1


def one_bit_extracted_set(number: int) -> Set[int]:
    """Return subsets of the ensemble with one bit less.

    :param number:
    :return: subsets in integer representation

    .. todo:: rename function
    """
    excluded = 1
    res: Set[int] = set()
    for i in range(size(number)):
        o_index = difference(number, excluded)
        excluded <<= 1
        if o_index != number:
            res.add(o_index)
    return res


def cardinal_1_subsets(number: int) -> Set[int]:
    """Return cardinal-1 subsets.

    :param number:
    :return:
    """
    res: Set[int] = set()
    n = 1
    for i in range(size(number)):
        if intersection(number, n) != 0:
            res.add(n)
        n <<= 1
    return res


def cardinal_subsets(number: int, cardinality_: int) -> Set[int]:
    """Return all subsets with given cardinality.

    :param number:
    :param cardinality_:
    :return:
    """
    res: Set[int] = set()
    sets = subsets(number)
    for s in sets:
        if cardinality(s) == cardinality_:
            res.add(s)
    return res


def proper_subsets(number: int) -> Set[int]:
    """Return proper subsets of given number.

    :param number:
    :return: subsets in integer representation
    """
    res: Set[int] = set()
    s = one_bit_extracted_set(number)
    res = set.union(res, s)
    for n in s:
        res = set.union(res, proper_subsets(n))
    return res


def subsets(number: int) -> Set[int]:
    """Return subsets of given number.

    :param number:
    :return: subsets in integer representation
    """
    return set.union({number}, proper_subsets(number))


def complement_subsets(number: int, set_size: int = None) -> Set[int]:
    """Return subsets of complement set.

    :param number:
    :param set_size: size of the ensemble
    :return: subsets in integer representation
    """
    return subsets(complement(number, set_size))
