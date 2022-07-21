# mypy: ignore-errors
"""This module contains set functions utilities.

.. todo::
    check and harmonize floating-point accuracy and tolerance across source
    and test code
"""
from math import factorial, isclose
from typing import List

import numpy as np

from .aliases import NumericValue
from .sets import (
    cardinal_1_subsets,
    cardinal_subsets,
    cardinality,
    difference,
    intersection,
    one_bit_extracted_set,
    size,
    subsets,
    union,
)


def is_powerset_function(set_function: List[NumericValue]) -> bool:
    """Check if the set function is defined on a power set.

    Check length of `set_function` is a power of `2`.

    :param set_function:
    :return:
    """
    return len(set_function) == 2 ** np.log2(len(set_function))


def is_game(set_function: List[NumericValue]) -> bool:
    """Check if the set function is a game.

    Check `set_function` is a power set function, and that
    :math:`\\nu(\\emptyset) = 0`

    :param set_function:
    :return:
    """
    return is_powerset_function(set_function) and set_function[0] == 0


def is_capacity(set_function: List[NumericValue]) -> bool:
    """Check if the set function is a capacity.

    Check `set_function` is a game, and that
    :math:`\\forall S, T \\subseteq N, S \\subseteq T \\Rightarrow \\nu(S) \\leq \\nu(T)`

    :param set_function:
    :return:

    .. note:: use :func:`math.isclose` to add tolerance
    """  # noqa E503
    if not is_game(set_function):
        return False
    for index in range(len(set_function)):
        # Compare capacity to capacity obtained by removing any one part
        for o_index in one_bit_extracted_set(index):
            if set_function[index] < set_function[o_index] and not isclose(
                set_function[index], set_function[o_index]
            ):
                return False
    return True


def is_normal_capacity(set_function: List[NumericValue]) -> bool:
    """Check if the set function is a normalized capacity.

    Check `set_function` is a capacity, and that :math:`\\nu(N) = 1`

    :param set_function:
    :return:

    .. note:: use :func:`math.isclose` to add tolerance
    """
    return is_capacity(set_function) and isclose(set_function[-1], 1)


def is_additive(set_function: List[NumericValue]) -> bool:
    """Check if the set function is additive or not.

    Check that
    :math:`\\forall S, T \\subseteq N, S \\cap T = \\emptyset, \\mu(S \\cup T) = \\mu(S) + \\mu(T)`

    :param set_function:
    :return:

    .. note:: use :func:`math.isclose` to add tolerance
    """  # noqa E503
    for s in range(len(set_function)):
        for t in range(s, len(set_function)):
            if intersection(s, t) != 0:
                continue
            if not isclose(
                set_function[union(s, t)], set_function[s] + set_function[t]
            ):
                return False
    return True


def is_k_additive(mobius: List[NumericValue], k: int) -> bool:
    """Check if möbius is of a k-additive capacity.

    :param mobius:
    :param k:
    :return:

    .. note:: use :func:`math.isclose` to add absolute tolerance

    .. todo:: uniformize absolute tolerance in whole package
    """
    for size_t in range(k + 1, size(len(mobius) - 1) + 1):
        for t in cardinal_subsets(len(mobius) - 1, size_t):
            if not isclose(mobius[t], 0, abs_tol=1e-9):
                return False
    for t in cardinal_subsets(len(mobius) - 1, k):
        if not isclose(mobius[t], 0, abs_tol=1e-9):
            return True
    return False


def is_cardinality_based(set_function: List[NumericValue]) -> bool:
    """Check if the set function is cardinality-based.

    Check that :math:`\\forall T \\subseteq N, \\mu(T)` only depends on `T`
    cardinality.

    :param set_function:
    :return:

    .. note:: use :func:`math.isclose` to add tolerance
    """
    cardinal_values = []
    n = 0
    while n < len(set_function):
        cardinal_values.append(set_function[n])
        n = n << 1 | 1
    for t in range(len(set_function)):
        if not isclose(set_function[t], cardinal_values[cardinality(t)]):
            return False
    return True


def is_mobius_capacity(set_function: List[NumericValue]) -> bool:
    """Check if the set function is a möbius capacity.

    Check `set_function` is a game, and that
    :math:`\\sum_{\\stackrel{T \\subseteq S}{T \\ni i}} m(T) \\geq 0, \\forall S \\subseteq N, \\forall i \\in S`

    :param set_function:
    :return:

    .. note:: use :func:`math.isclose` to add absolute tolerance

    .. todo:: uniformize absolute tolerance in whole package
    """  # noqa E503
    if not is_game(set_function):
        return False
    for s in range(len(set_function)):
        for i in cardinal_1_subsets(s):
            res = 0.0
            for t in subsets(s):
                if intersection(t, i) != 0:
                    res += set_function[t]
            if res < 0 and not isclose(res, 0, abs_tol=1e-9):
                return False
    return True


def is_normal_capacity_mobius(set_function: List[NumericValue]) -> bool:
    """Check if the set function is a normalized möbius capacity.

    Check `set_function` is a möbius capacity, and that
    :math:`\\sum_{T \\subseteq N} m(T) = 1`

    :param set_function:
    :return:

    .. note:: use :func:`math.isclose` to add tolerance
    """
    return is_mobius_capacity(set_function) and isclose(sum(set_function), 1)


def shapley_capacity(capacity: List[NumericValue]) -> List[NumericValue]:
    """Return list of Shapley values computed using capacity.

    :param capacity:
    :return:

    .. note:: Formula is based on :cite:p:`grabisch2008review`.
    """
    return [
        sum(
            factorial(size(len(capacity) - 1) - cardinality(t) - 1)
            * factorial(cardinality(t))
            * (capacity[union(t, i)] - capacity[t])
            / factorial(size(len(capacity) - 1))
            for t in subsets(difference(len(capacity) - 1, i))
        )
        for i in sorted(cardinal_1_subsets(len(capacity) - 1))
    ]


def shapley_mobius(mobius: List[NumericValue]) -> List[NumericValue]:
    """Return list of Shapley values computed using möbius capacity.

    :param mobius:
    :return:

    .. note:: Formula is based on :cite:p:`grabisch2008review`.
    """
    return [
        sum(
            mobius[union(t, i)] / (cardinality(t) + 1)
            for t in subsets(difference(len(mobius) - 1, i))
        )
        for i in sorted(cardinal_1_subsets(len(mobius) - 1))
    ]


def interaction_index_capacity(
    capacity: List[NumericValue],
) -> List[List[NumericValue]]:
    """Return interaction index matrix computed using capacity.

    :param capacity:
    :return:

    .. note:: Formula is based on :cite:p:`grabisch2008review`.
    """
    return [
        [
            float("nan")
            if i == j
            else sum(
                factorial(size(len(capacity) - 1) - cardinality(t) - 2)
                * factorial(cardinality(t))
                * (
                    capacity[union(t, union(i, j))]
                    - capacity[union(t, i)]
                    - capacity[union(t, j)]
                    + capacity[t]
                )
                / factorial(size(len(capacity) - 1) - 1)
                for t in subsets(difference(len(capacity) - 1, union(i, j)))
            )
            for j in sorted(cardinal_1_subsets(len(capacity) - 1))
        ]
        for i in sorted(cardinal_1_subsets(len(capacity) - 1))
    ]


def interaction_index_mobius(
    mobius: List[NumericValue],
) -> List[List[NumericValue]]:
    """Return interaction index matrix computed using möbius.

    :param mobius:
    :return:

    .. note:: Formula is based on :cite:p:`grabisch2008review`.
    """
    return [
        [
            float("nan")
            if i == j
            else sum(
                mobius[union(t, union(i, j))] / (cardinality(t) + 1)
                for t in subsets(difference(len(mobius) - 1, union(i, j)))
            )
            for j in sorted(cardinal_1_subsets(len(mobius) - 1))
        ]
        for i in sorted(cardinal_1_subsets(len(mobius) - 1))
    ]


def uniform_capacity(size: int) -> List[NumericValue]:
    """Return uniform capacity of given set size.

    The uniform capacity on an ensemble `N` of size `size` is given by:

    .. math::

        \\mu^*(T) = t/n, \\forall T \\subseteq N

    :param size:
    :return:

    .. note:: Formula is based on :cite:p:`grabisch2008review`.
    """
    return [cardinality(t) / size for t in range(2 ** size)]


def mobius_transform(set_function: List[NumericValue]) -> List[NumericValue]:
    """Return möbius transform of a set function.

    :param set_function:
    :return:

    .. note:: Formula is based on :cite:p:`grabisch2008review`.
    """
    return [
        sum(
            ((-1) ** (cardinality(s) - cardinality(t))) * set_function[t]
            for t in subsets(s)
        )
        for s in range(len(set_function))
    ]


def mobius_transform_numpy(
    set_function: List[NumericValue],
) -> List[NumericValue]:
    """Return möbius transform of a set function.

    Computed using numpy.

    :param set_function:
    :return:

    .. note:: included in case this form of function is needed elsewhere

    .. warning:: less efficient and precise than :func:`mobius_transform_fast`

    .. note:: Formula is based on :cite:p:`grabisch2008review`.
    """
    v = np.array(set_function)
    card = np.array([cardinality(v) for v in range(len(set_function))])
    matrix = card.reshape(len(card), 1).dot(np.ones((1, len(card)))) - np.ones(
        (len(card), 1)
    ).dot(card.reshape(1, len(card)))
    matrix = (-1) ** matrix
    matrix *= np.ones((len(v), 1), dtype=np.int).dot(v.reshape(1, len(v)))
    all_sets = np.ones((len(v), 1), dtype=np.int).dot(
        np.arange(len(v)).reshape(1, len(v))
    )
    t = all_sets.transpose()
    subs = np.zeros_like(all_sets)
    subs[np.where(all_sets & t == all_sets)] = 1
    matrix *= subs
    return np.sum(
        matrix,
        axis=1,
    ).tolist()


def mobius_transform_fast(
    set_function: List[NumericValue],
) -> List[NumericValue]:
    """Return möbius transform of a set function.

    Computed using fast method.

    :param set_function:
    :return:

    .. note::
        * Formula is based on :cite:p:`grabisch2008review`.
        * Implementation is based on :cite:p:`bigaret2017mcdar`.
    """
    n = size(len(set_function) - 1)
    res = [v for v in set_function]
    for i in range(1, n + 1):
        for j in range(1, 1 << i, 2):
            for k in range(1 << (n - i)):
                res[j * (1 << (n - i)) + k] += (
                    -1 * res[(j - 1) * (1 << (n - i)) + k]
                )
    return res


def inverse_mobius_transform(mobius: List[NumericValue]) -> List[NumericValue]:
    """Return set function from it möbius transform.

    :param mobius:
    :return:

    .. note:: Formula is based on :cite:p:`grabisch2008review`.
    """
    return [sum(mobius[s] for s in subsets(t)) for t in range(len(mobius))]
