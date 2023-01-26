"""This module implements methods to compute concordance.

Implementation based on:
    * concordance: :cite:p:`DelVastoTerrientes15`,
    * concordance_reinforced: :cite:p:`Roy08`,
    * concordance_with_interactions: :cite:p:`Figueira09`.
"""
from enum import Enum
from typing import Any, Dict, Hashable, Optional, Tuple, Union

import pandas as pd

from ..core.aliases import NumericValue
from ..core.functions import Threshold
from ..core.scales import PreferenceDirection, QuantitativeScale
from . import exceptions
from ._validation import (
    _both_values_in_scale,
    _check_df_index,
    _consistent_criteria_names,
    _get_threshold_values,
    _inverse_values,
    _reinforcement_factors_vals,
    _weights_proper_vals,
)


def concordance_marginal(
    a_value: NumericValue,
    b_value: NumericValue,
    scale: QuantitativeScale,
    indifference_threshold: Threshold,
    preference_threshold: Threshold,
    inverse: bool = False,
) -> NumericValue:
    """Computes marginal concordance index: :math:`c(a\\_value, b\\_value) \\in [0, 1]`,
    which represents a degree to which criterion supports
    the hypothesis about outranking a over b :math:`(aSb)`.

    :param a_value: alternative's performance value on one criterion
    :param b_value: alternative's performance value on the same criterion as `a_value`
    :param scale: criterion's scale with specified preference direction
    :param indifference_threshold: criterion's indifference threshold
    :param preference_threshold: criterion's preference threshold
    :param inverse: if ``True``, then function will calculate `c(b_value, a_value)`
        with opposite scale preference direction, defaults to ``False``

    :raises exceptions.WrongThresholdValueError: if thresholds don't meet a condition:
    .. math::
        0 \\le indifference\\_threshold(value) \\le preference\\_threshold(value)

    :return: marginal concordance index, value from the [0, 1] interval
    """
    _both_values_in_scale(a_value, b_value, scale)
    a_value, b_value, scale = _inverse_values(a_value, b_value, scale, inverse)

    q_a, p_a = _get_threshold_values(
        a_value,
        indifference_threshold=indifference_threshold,
        preference_threshold=preference_threshold,
    )

    if not 0 <= q_a <= p_a:
        raise exceptions.WrongThresholdValueError(
            "Threshold values must meet a condition: "
            "0 <= indifference_threshold <= preference_threshold, but got "
            f"0 <= {q_a} <= {p_a}"
        )

    if scale.preference_direction == PreferenceDirection.MIN:
        if a_value - b_value <= q_a:
            return 1.0
        elif a_value - b_value > p_a:
            return 0.0
        return (b_value - a_value + p_a) / (p_a - q_a)

    if b_value - a_value <= q_a:
        return 1.0
    elif b_value - a_value > p_a:
        return 0.0
    return (p_a - (b_value - a_value)) / (p_a - q_a)


def concordance_comprehensive(
    a_values: Union[Dict[Any, NumericValue], pd.Series],
    b_values: Union[Dict[Any, NumericValue], pd.Series],
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    weights: Union[Dict[Any, NumericValue], pd.Series],
    indifference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    inverse: bool = False,
    **kwargs,
) -> NumericValue:
    """Computes comprehensive discordance index :math:`C^S(a, b) \\in [0, 1]`,
    which represents a degree to which alternative `a` outranks `b`.

    :param a_values: alternative's performance on all its criteria
    :param b_values: alternative's performance on all its criteria
    :param scales: all criteria's scales with specified preference direction
    :param weights: all criteria's weights, :math:`weight\\_value > 0`
    :param indifference_thresholds: all criteria's indifference thresholds
    :param preference_thresholds: all criteria's preference thresholds
    :param inverse: if ``True``, then function will calculate :math:`C^S(b, a)`
        with opposite scales preference direction, defaults to ``False``

    :return: comprehensive concordance index, value from the [0, 1] interval
    """
    if "validated" not in kwargs:
        _consistent_criteria_names(
            a_values=a_values,
            b_values=b_values,
            scales=scales,
            weights=weights,
            indifference_thresholds=indifference_thresholds,
            preference_thresholds=preference_thresholds,
        )
        _weights_proper_vals(weights)

    return sum(
        [
            weights[criterion_name]
            * concordance_marginal(
                a_values[criterion_name],
                b_values[criterion_name],
                scales[criterion_name],
                indifference_thresholds[criterion_name],
                preference_thresholds[criterion_name],
                inverse,
            )
            for criterion_name in a_values.keys()
        ]
    ) / sum(weights.values() if isinstance(weights, dict) else weights.values)


def concordance(
    alternatives_perform: pd.DataFrame,
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    weights: Union[Dict[Any, NumericValue], pd.Series],
    indifference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    profiles_perform: Optional[pd.DataFrame] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Computes comprehensive concordance indices :math:`C^S(a, b) \\in [0, 1]`.
    Comparison is possible for alternatives or between alternatives and
    profiles, depending on the `profiles_perform` argument value.

    :param alternatives_perform: performance table of alternatives
    :param scales: all criteria's scales with specified preference direction
    :param weights: all criteria's weights, :math:`weight\\_value > 0`
    :param indifference_thresholds: all criteria's indifference thresholds
    :param preference_thresholds: all criteria's preference thresholds
    :param profiles_perform: if provided, indices would be computed
        between alternatives from `alternatives_perform` and from this argument,
        defaults to ``None``

    :return:
        * if `profiles_perform` argument is set to ``None``, the function will
          return a single `pandas.DataFrame` object with :math:`C^S(a, b)` indices
          for all alternatives pairs
        * otherwise, the function will return a ``tuple`` object with two
          `pandas.DataFrame` objects inside, where the first one contains
          :math:`C^S(alternative_i, profile_j)` indices, and the second one
          :math:`C^S(profile_j, alternative_i)`, respectively
    """
    _consistent_criteria_names(
        alternatives_perform=alternatives_perform,
        scales=scales,
        weights=weights,
        indifference_thresholds=indifference_thresholds,
        preference_thresholds=preference_thresholds,
        profiles_perform=profiles_perform,
    )
    _check_df_index(alternatives_perform, index_type="alternatives")
    _check_df_index(profiles_perform, index_type="profiles")
    _weights_proper_vals(weights)
    if profiles_perform is not None:
        return pd.DataFrame(
            [
                [
                    concordance_comprehensive(
                        alternatives_perform.loc[alt_name],
                        profiles_perform.loc[prof_name],
                        scales,
                        weights,
                        indifference_thresholds,
                        preference_thresholds,
                        validated=True,
                    )
                    for prof_name in profiles_perform.index.values
                ]
                for alt_name in alternatives_perform.index.values
            ],
            index=alternatives_perform.index,
            columns=profiles_perform.index,
        ), pd.DataFrame(
            [
                [
                    concordance_comprehensive(
                        profiles_perform.loc[prof_name],
                        alternatives_perform.loc[alt_name],
                        scales,
                        weights,
                        indifference_thresholds,
                        preference_thresholds,
                        validated=True,
                    )
                    for alt_name in alternatives_perform.index.values
                ]
                for prof_name in profiles_perform.index.values
            ],
            index=profiles_perform.index,
            columns=alternatives_perform.index,
        )

    return pd.DataFrame(
        [
            [
                concordance_comprehensive(
                    alternatives_perform.loc[alt_name_a],
                    alternatives_perform.loc[alt_name_b],
                    scales,
                    weights,
                    indifference_thresholds,
                    preference_thresholds,
                    validated=True,
                )
                for alt_name_b in alternatives_perform.index.values
            ]
            for alt_name_a in alternatives_perform.index.values
        ],
        index=alternatives_perform.index,
        columns=alternatives_perform.index,
    )


def is_reinforcement_occur(
    a_value: NumericValue,
    b_value: NumericValue,
    scale: QuantitativeScale,
    preference_threshold: Threshold,
    reinforcement_threshold: Threshold,
) -> bool:
    """Checks if the reinforcement occurs when comparing
    two alternatives on a criterion, i.e. if the `a`
    alternative is better than `b` on specified criterion
    by more than the `reinforcement_threshold` value.

    :param a_value: alternative's performance value on one criterion
    :param b_value: alternative's performance value on the same criterion as `a_value`
    :param scale: criterion's scale with specified preference direction
    :param preference_threshold: criterion's preference threshold
    :param reinforcement_threshold: criterion's reinforcement threshold

    :raises exceptions.WrongThresholdValueError: if thresholds don't meet a condition:
    .. math::
        0 \\le preference\\_threshold(b\\_value) < reinforcement\\_threshold(b\\_value)

    :return: ``True``, if reinforcement condition is fulfilled, ``False`` otherwise
    """
    _both_values_in_scale(a_value, b_value, scale)
    threshold_value, preference_value = _get_threshold_values(
        b_value,
        reinforcement_threshold=reinforcement_threshold,
        preference_threshold=preference_threshold,
    )
    if not 0 <= preference_value < threshold_value:
        raise exceptions.WrongThresholdValueError(
            "Threshold values must meet a condition: "
            "0 <= preference_threshold < reinforcement_threshold, but got "
            f"0 <= {preference_value} < {threshold_value}"
        )

    return (
        a_value - b_value > threshold_value
        if scale.preference_direction == PreferenceDirection.MAX
        else b_value - a_value > threshold_value
    )


def _get_reinforced_criteria(
    a_values: Union[Dict[Any, NumericValue], pd.Series],
    b_values: Union[Dict[Any, NumericValue], pd.Series],
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    reinforced_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
) -> pd.Series:
    """Return a series with information about reinforcement occurrence
    between performance values from `a_values` and `b_values` on all their
    criteria.

    For each criterion, if no reinforcement threshold is specified, the
    reinforcement is set to ``False``.
    """
    result = pd.Series([False] * len(a_values), index=list(a_values.keys()))
    for criterion_name in a_values.keys():
        threshold = reinforced_thresholds[criterion_name]
        if isinstance(threshold, Threshold):
            result[criterion_name] = is_reinforcement_occur(
                a_values[criterion_name],
                b_values[criterion_name],
                scales[criterion_name],
                preference_thresholds[criterion_name],
                threshold,
            )
    return result


def concordance_reinforced_comprehensive(
    a_values: Union[Dict[Any, NumericValue], pd.Series],
    b_values: Union[Dict[Any, NumericValue], pd.Series],
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    weights: Union[Dict[Any, NumericValue], pd.Series],
    indifference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    reinforced_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
    reinforcement_factors: Union[Dict[Any, Optional[NumericValue]], pd.Series],
    inverse: bool = False,
    **kwargs,
) -> NumericValue:
    """Computes concordance index with reinforcement
    :math:`C^{RP}(a, b) \\in [0, 1]`, which represents a degree
    to which alternative `a` outranks `b` with considering the
    reinforcement effect.

    :param a_values: alternative's performance on all its criteria
    :param b_values: alternative's performance on all its criteria
    :param scales: all criteria's scales with specified preference direction
    :param weights: all criteria's weights, :math:`weight\\_value > 0`
    :param indifference_thresholds: all criteria's indifference thresholds
    :param preference_thresholds: all criteria's preference thresholds
    :param reinforced_thresholds: all criteria's reinforcement thresholds;
        for each criterion, providing a ``Threshold`` is optional and it
        can be set to ``None``
    :param reinforcement_factors: if criterion has specified reinforcement
        threshold, the reinforcement factor :math:`\\omega_j > 1`
        must be set as well; otherwise, it should be ``None``
    :param inverse: if ``True``, then function will calculate :math:`C^S(b, a)`
        with opposite scales preference direction, defaults to ``False``

    :raises ValueError: if reinforcement threshold for a criterion was
        specified, but its reinforcement factor is ``None``

    :return: reinforced concordance index, value from the [0, 1] interval
    """
    if "validated" not in kwargs:
        _consistent_criteria_names(
            a_values=a_values,
            b_values=b_values,
            scales=scales,
            weights=weights,
            indifference_thresholds=indifference_thresholds,
            preference_thresholds=preference_thresholds,
            reinforced_thresholds=reinforced_thresholds,
            reinforcement_factors=reinforcement_factors,
        )
        _weights_proper_vals(weights)
        _reinforcement_factors_vals(reinforcement_factors)

    reinforce_occur = _get_reinforced_criteria(
        a_values, b_values, scales, preference_thresholds, reinforced_thresholds
    )
    sum_weights_reinforced: NumericValue = 0
    sum_weights_not_reinforced: NumericValue = 0
    sum_concordances_not_reinforced: NumericValue = 0
    for criterion_name in reinforce_occur.index.values:
        if reinforce_occur[criterion_name]:
            factor = reinforcement_factors[criterion_name]
            if factor is None:
                raise ValueError(
                    "Reinforce threshold was provided, but there's missing factor "
                    f"on {criterion_name} criterion."
                )
            else:
                sum_weights_reinforced += weights[criterion_name] * factor
        else:
            sum_weights_not_reinforced += weights[criterion_name]
            sum_concordances_not_reinforced += weights[criterion_name] * concordance_marginal(
                a_values[criterion_name],
                b_values[criterion_name],
                scales[criterion_name],
                indifference_thresholds[criterion_name],
                preference_thresholds[criterion_name],
                inverse,
            )

    return (sum_concordances_not_reinforced + sum_weights_reinforced) / (
        sum_weights_reinforced + sum_weights_not_reinforced
    )


def concordance_reinforced(
    alternatives_perform: pd.DataFrame,
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    weights: Union[Dict[Any, NumericValue], pd.Series],
    indifference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    reinforced_thresholds: Union[Dict[Any, Optional[Threshold]], pd.Series],
    reinforcement_factors: Union[Dict[Any, Optional[NumericValue]], pd.Series],
    profiles_perform: Optional[pd.DataFrame] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Computes reinforced concordance indices :math:`C^{RP}(a, b) \\in [0, 1]`.
    Comparison is possible for alternatives or between alternatives and
    profiles, depending on the `profiles_perform` argument value.

    :param alternatives_perform: performance table of alternatives
    :param scales: all criteria's scales with specified preference direction
    :param weights: all criteria's weights, :math:`weight\\_value > 0`
    :param indifference_thresholds: all criteria's indifference thresholds
    :param preference_thresholds: all criteria's preference thresholds
    :param reinforced_thresholds: all criteria's reinforcement thresholds;
        for each criterion, providing a ``Threshold`` is optional and it
        can be set to ``None``
    :param reinforcement_factors: if criterion has specified reinforcement
        threshold, the reinforcement factor :math:`\\omega_j > 1`
        must be set as well; otherwise, it should be ``None``
    :param profiles_perform: if provided, indices would be computed
        between alternatives from `alternatives_perform` and from this argument,
        defaults to ``None``

    :return:
        * if `profiles_perform` argument is set to ``None``, the function will
          return a single `pandas.DataFrame` object with :math:`C^{RP}(a, b)` indices
          for all alternatives pairs
        * otherwise, the function will return a ``tuple`` object with two
          `pandas.DataFrame` objects inside, where the first one contains
          :math:`C^{RP}(alternative_i, profile_j)` indices, and the second one
          :math:`C^{RP}(profile_j, alternative_i)`, respectively
    """
    _consistent_criteria_names(
        alternatives_perform=alternatives_perform,
        scales=scales,
        weights=weights,
        indifference_thresholds=indifference_thresholds,
        preference_thresholds=preference_thresholds,
        reinforced_thresholds=reinforced_thresholds,
        reinforcement_factors=reinforcement_factors,
        profiles_perform=profiles_perform,
    )
    _check_df_index(alternatives_perform, index_type="alternatives")
    _check_df_index(profiles_perform, index_type="profiles")
    _weights_proper_vals(weights)
    _reinforcement_factors_vals(reinforcement_factors)
    if profiles_perform is not None:
        return pd.DataFrame(
            [
                [
                    concordance_reinforced_comprehensive(
                        alternatives_perform.loc[alt_name],
                        profiles_perform.loc[prof_name],
                        scales,
                        weights,
                        indifference_thresholds,
                        preference_thresholds,
                        reinforced_thresholds,
                        reinforcement_factors,
                        validated=True,
                    )
                    for prof_name in profiles_perform.index.values
                ]
                for alt_name in alternatives_perform.index.values
            ],
            index=alternatives_perform.index,
            columns=profiles_perform.index,
        ), pd.DataFrame(
            [
                [
                    concordance_reinforced_comprehensive(
                        profiles_perform.loc[prof_name],
                        alternatives_perform.loc[alt_name],
                        scales,
                        weights,
                        indifference_thresholds,
                        preference_thresholds,
                        reinforced_thresholds,
                        reinforcement_factors,
                        validated=True,
                    )
                    for alt_name in alternatives_perform.index.values
                ]
                for prof_name in profiles_perform.index.values
            ],
            index=profiles_perform.index,
            columns=alternatives_perform.index,
        )

    return pd.DataFrame(
        [
            [
                concordance_reinforced_comprehensive(
                    alternatives_perform.loc[alt_name_a],
                    alternatives_perform.loc[alt_name_b],
                    scales,
                    weights,
                    indifference_thresholds,
                    preference_thresholds,
                    reinforced_thresholds,
                    reinforcement_factors,
                    validated=True,
                )
                for alt_name_b in alternatives_perform.index.values
            ]
            for alt_name_a in alternatives_perform.index.values
        ],
        index=alternatives_perform.index,
        columns=alternatives_perform.index,
    )


class InteractionType(Enum):
    """An `Enum` class to specify type of interaction during the
    concordance with interactions computation.
    """

    MS = "Mutual Strengthening"
    MW = "Mutual Weakening"
    A = "Antagonistic"


class FunctionType(Enum):
    """An `Enum` class to specify type of function during the
    concordance with interactions computation.

    The function will multiply the pair of concordance indices
    or choose a lower value.
    """

    MUL = "Multiplication"
    MIN = "Minimisation"


class Interaction:
    """Class type for interaction representation, used in
    concordance with interactions computation.

    :param interaction_type: type of the interaction,
        one of the `InteractionType` option
    :param factor: interaction factor, value used for
        multiply a concordance value; for the
        mutual weakening effect, the value must be negative,
        otherwise positive
    """

    def __init__(
        self,
        interaction_type: InteractionType,
        factor: NumericValue,
    ):
        self.interaction = interaction_type
        self.factor = factor

    def __repr__(self) -> str:
        """Return instance representation string."""
        return f"{self.interaction.value} |" f" {self.factor} "


def _check_interaction_factor_value(interaction: Interaction) -> None:
    """Checks if interaction factor has proper value,
    depending on its type.

    :raises exception.WrongFactorValueError (ValueError):
        * if factor has the wrong value

    :raises TypeError:
        * if `interaction` has wrong type
        * if `interaction.factor` is not numeric
    """
    try:
        if interaction.interaction == InteractionType.MW and not interaction.factor < 0:
            raise exceptions.WrongFactorValueError(
                "Interaction factor for mutual weakening effect should be "
                f"negative, but got {interaction.factor} instead."
            )
        elif interaction.interaction != InteractionType.MW and not interaction.factor > 0:
            raise exceptions.WrongFactorValueError(
                "Interaction factor for mutual strengthening and antagonistic effect "
                f"should be positive, but got {interaction.factor} instead."
            )
    except AttributeError as exc:
        raise TypeError(
            f"Wrong interaction type. Expected {Interaction.__name__}, but "
            f"got {type(interaction).__name__} instead."
        ) from exc
    except TypeError as exc:
        exc.args = (
            "Wrong interaction factor type. Expected numeric, but got "
            f"{type(interaction.factor).__name__} instead.",
        )
        raise


def _positive_net_balance(
    criterion_name: Hashable,
    weight: NumericValue,
    criterion_interactions: pd.Series,
) -> None:
    """Checks if positive net balance condition is fulfilled.

    :raises exceptions.PositiveNetBalanceError (ValueError):
        * if pnb <= 0
    """
    if (
        weight
        - sum(
            [
                abs(x.factor)
                for x in criterion_interactions
                if isinstance(x, Interaction)
                and x.interaction in (InteractionType.MW, InteractionType.A)
            ]
        )
        <= 0
    ):
        raise exceptions.PositiveNetBalanceError(
            f"Positive net balance condition is not fulfilled on a {criterion_name} criterion."
        )


def concordance_with_interactions_marginal(
    a_values: Union[Dict[Any, NumericValue], pd.Series],
    b_values: Union[Dict[Any, NumericValue], pd.Series],
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    weights: Union[Dict[Any, NumericValue], pd.Series],
    indifference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    interactions: pd.DataFrame,
    function_type: FunctionType = FunctionType.MIN,
    **kwargs,
) -> NumericValue:
    """Computes concordance index with interactions
    :math:`C^{INT}(a, b) \\in [0, 1]`, which represents a degree
    to which alternative `a` outranks `b` with considering the
    interactions effects on criteria.

    :param a_values: alternative's performance on all its criteria
    :param b_values: alternative's performance on all its criteria
    :param scales: all criteria's scales with specified preference direction
    :param weights: all criteria's weights, :math:`weight\\_value > 0`
    :param indifference_thresholds: all criteria's indifference thresholds
    :param preference_thresholds: all criteria's preference thresholds
    :param interactions: table with information about the interactions
        between criteria; both rows and columns must be indexed
        with criteria names; for pairs without an interaction, value
        should be set to ``None``
    :param function_type: specifies the function used during the
        interaction value calculation, defaults to FunctionType.MIN

    :return: concordance with interactions index, value from the [0, 1] interval
    """
    if "validated" not in kwargs:
        _consistent_criteria_names(
            a_values=a_values,
            b_values=b_values,
            scales=scales,
            weights=weights,
            indifference_thresholds=indifference_thresholds,
            preference_thresholds=preference_thresholds,
            interactions=interactions,
        )
        _check_df_index(interactions, index_type="criteria")
        _weights_proper_vals(weights)

        if set(interactions.index.values) != set(interactions.columns.values):
            raise exceptions.InconsistentIndexNamesError(
                "Interaction DataFrame index should contain the same values set as " "its columns."
            )

        if not isinstance(function_type, FunctionType):
            raise TypeError(
                f"Wrong FunctionType argument. Expected {FunctionType.__name__}, "
                f"but got {type(function_type).__name__} instead."
            )

    mutual, antagonistic = [], []
    marginal_concordances = pd.Series(
        [
            concordance_marginal(
                a_values[criterion_name],
                b_values[criterion_name],
                scales[criterion_name],
                indifference_thresholds[criterion_name],
                preference_thresholds[criterion_name],
            )
            for criterion_name in a_values.keys()
        ],
        index=list(a_values.keys()),
    )
    interaction_value = (
        lambda c_i, c_j: min(c_i, c_j) if function_type == FunctionType.MIN else c_i * c_j
    )

    for i in interactions.index.values:
        for j in interactions.columns.values:
            if interactions[j][i] is not None:
                if j == i:
                    raise exceptions.WrongInteractionError(
                        f"Criterion {i} cannot interact with itself."
                    )

                c_i = marginal_concordances[i]
                c_j = marginal_concordances[j]

                _check_interaction_factor_value(interactions[j][i])
                _positive_net_balance(
                    criterion_name=j, weight=weights[j], criterion_interactions=interactions.loc[j]
                )
                if interactions[j][i].interaction in (
                    InteractionType.MS,
                    InteractionType.MW,
                ):
                    mutual.append(interaction_value(c_i, c_j) * interactions[j][i].factor)

                elif interactions[j][i].interaction == InteractionType.A:
                    antagonistic.append(
                        interaction_value(
                            c_i,
                            concordance_marginal(
                                b_values[j],
                                a_values[j],
                                scales[j],
                                indifference_thresholds[j],
                                preference_thresholds[j],
                            ),
                        )
                        * interactions[j][i].factor
                    )
                else:
                    raise exceptions.WrongInteractionError(
                        f"Interaction type should has a{InteractionType.__name__} type, "
                        f"but got {type(interactions[j][i].interaction).__name__} instead."
                    )

    interaction_sum = sum(mutual) - sum(antagonistic)
    return (
        sum(
            [
                weights[criterion_name] * marginal_concordances[criterion_name]
                for criterion_name in a_values.keys()
            ]
        )
        + interaction_sum
    ) / (sum(weights) + interaction_sum)


def concordance_with_interactions(
    alternatives_perform: pd.DataFrame,
    scales: Union[Dict[Any, QuantitativeScale], pd.Series],
    weights: Union[Dict[Any, NumericValue], pd.Series],
    indifference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    preference_thresholds: Union[Dict[Any, Threshold], pd.Series],
    interactions: pd.DataFrame,
    function_type: FunctionType = FunctionType.MIN,
    profiles_perform: Optional[pd.DataFrame] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Computes concordance with interactions indices
    :math:`C^{INT}(a, b) \\in [0, 1]`. Comparison is possible for
    alternatives or between alternatives and profiles, depending
    on the `profiles_perform` argument value.

    :param alternatives_perform: performance table of alternatives, where single
        row is a performance of one alternative (which means that one column
        represents single criterion values of all alternatives)
    :param scales: all criteria's scales with specified preference direction
    :param weights: all criteria's weights, :math:`weight\\_value > 0`
    :param indifference_thresholds: all criteria's indifference thresholds
    :param preference_thresholds: all criteria's preference thresholds
    :param interactions: table with information about the interactions
        between criteria; both rows and columns must be indexed
        with criteria names; for pairs without an interaction, value
        should be set to ``None``
    :param function_type: specifies the function used during the
        interaction value calculation, defaults to FunctionType.MIN
    :param profiles_perform: if provided, indices would be computed
        between alternatives from `alternatives_perform` and from this argument,
        defaults to ``None``

    :return:
        * if `profiles_perform` argument is set to ``None``, the function will
          return a single `pandas.DataFrame` object with :math:`C^{RP}(a, b)` indices
          for all alternatives pairs
        * otherwise, the function will return a ``tuple`` object with two
          `pandas.DataFrame` objects inside, where the first one contains
          :math:`C^{RP}(alternative_i, profile_j)` indices, and the second one
          :math:`C^{RP}(profile_j, alternative_i)`, respectively
    """
    _consistent_criteria_names(
        alternatives_perform=alternatives_perform,
        scales=scales,
        weights=weights,
        indifference_thresholds=indifference_thresholds,
        preference_thresholds=preference_thresholds,
        interactions=interactions,
        profiles_perform=profiles_perform,
    )
    _check_df_index(interactions, index_type="criteria")
    _weights_proper_vals(weights)
    if set(interactions.index.values) != set(interactions.columns.values):
        raise exceptions.InconsistentIndexNamesError(
            "Interaction DataFrame index should contain the same values set as " "its columns."
        )
    if not isinstance(function_type, FunctionType):
        raise TypeError(
            f"Wrong FunctionType argument. Expected {FunctionType.__name__}, "
            f"but got {type(function_type).__name__} instead."
        )

    if profiles_perform is not None:
        return pd.DataFrame(
            [
                [
                    concordance_with_interactions_marginal(
                        alternatives_perform.loc[alt_name],
                        profiles_perform.loc[prof_name],
                        scales,
                        weights,
                        indifference_thresholds,
                        preference_thresholds,
                        interactions,
                        function_type,
                        validated=True,
                    )
                    for prof_name in profiles_perform.index.values
                ]
                for alt_name in alternatives_perform.index.values
            ],
            index=alternatives_perform.index,
            columns=profiles_perform.index,
        ), pd.DataFrame(
            [
                [
                    concordance_with_interactions_marginal(
                        profiles_perform.loc[prof_name],
                        alternatives_perform.loc[alt_name],
                        scales,
                        weights,
                        indifference_thresholds,
                        preference_thresholds,
                        interactions,
                        function_type,
                        validated=True,
                    )
                    for alt_name in alternatives_perform.index.values
                ]
                for prof_name in profiles_perform.index.values
            ],
            index=profiles_perform.index,
            columns=alternatives_perform.index,
        )
    return pd.DataFrame(
        [
            [
                concordance_with_interactions_marginal(
                    alternatives_perform.loc[alt_name_a],
                    alternatives_perform.loc[alt_name_b],
                    scales,
                    weights,
                    indifference_thresholds,
                    preference_thresholds,
                    interactions,
                    function_type,
                    validated=True,
                )
                for alt_name_b in alternatives_perform.index.values
            ]
            for alt_name_a in alternatives_perform.index.values
        ],
        index=alternatives_perform.index,
        columns=alternatives_perform.index,
    )
