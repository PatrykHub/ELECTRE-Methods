from ..core.scales import QuantitativeScale, PreferenceDirection
from ..core.aliases import NumericValue
from typing import List, Union, Tuple


def discordance_marginal_bin(a_value: NumericValue,
                             b_value: NumericValue,
                             scale: QuantitativeScale,
                             veto_threshold: List[NumericValue],
                             inverse: bool = False) -> int:
    """
    Calculates binary marginal discordance with veto threshold.
    :param a_value:
    :param b_value:
    :param scale:
    :param veto_threshold:
    :param inverse:
    :return:
    """
    try:
        if a_value not in scale or b_value not in scale:
            # TODO
            raise ValueError('')
    except TypeError as e:
        e.args = ('Both criteria values have to be numeric values.',)
        raise

    try:
        if inverse:
            a_value, b_value = b_value, a_value
            scale.preference_direction = PreferenceDirection.MIN \
                if scale.preference_direction == PreferenceDirection.MAX \
                else PreferenceDirection.MAX

        veto: NumericValue = veto_threshold[0] * a_value + veto_threshold[1]
        if scale.preference_direction == PreferenceDirection.MAX:
            return 1 if b_value - a_value >= veto else 0
        return 1 if a_value - b_value >= veto else 0
    except (IndexError, TypeError) as e:
        e.args = ('Veto threshold needs to be a two element list.',)
        raise
    except AttributeError as e:
        e.args = (f'Scale have to be a QuantitativeScale object, got {type(scale).__name__} instead.',)
        raise


def discordance_comprehensive_bin(a: List[NumericValue],
                                  b: List[NumericValue],
                                  scales: List[QuantitativeScale],
                                  veto_thresholds: List[List[NumericValue]],
                                  inverse: bool = False) -> int:
    """
    Calculates comprehensive binary discordance.
    :param a:
    :param b:
    :param scales:
    :param veto_thresholds:
    :param inverse:
    :return:
    """
    try:
        list_len: int = max(len(el) for el in (a, b, scales, veto_thresholds))
        return 1 \
            if 1 in [discordance_marginal_bin(a[i], b[i], scales[i], veto_thresholds[i], inverse)
                     for i in range(list_len)] \
            else 0
    except TypeError:
        # TODO ktorys arg nie jest lista
        pass
    except IndexError:
        # TODO rozna dlugosc list
        pass


def discordance_bin(alternatives_perform: List[List[NumericValue]],
                    scales: List[QuantitativeScale],
                    veto_thresholds: List[List[NumericValue]],
                    profiles_perform: List[List[NumericValue]] = None
                    ) -> Union[List[List[int]], Tuple]:
    """
    :param alternatives_perform:
    :param scales:
    :param veto_thresholds:
    :param profiles_perform:
    :return:
    """
    if profiles_perform is None:
        return [
            [
                discordance_comprehensive_bin(
                    alternatives_perform[i], alternatives_perform[j], scales, veto_thresholds
                ) for j in range(len(alternatives_perform[i]))
            ] for i in range(len(alternatives_perform))
        ]

    return [
        [
            discordance_comprehensive_bin(
                alternatives_perform[i], profiles_perform[j], scales, veto_thresholds
            ) for j in range(len(profiles_perform))
        ] for i in range(len(alternatives_perform))
    ], [
        [
            discordance_comprehensive_bin(
                profiles_perform[i], alternatives_perform[j], scales, veto_thresholds
            ) for j in range(len(alternatives_perform))
        ] for i in range(len(profiles_perform))
    ]

"""This module implements methods to compute discordance."""
from typing import List, Optional

from ..core.aliases import NumericValue
from ..core.scales import QuantitativeScale, PreferenceDirection
from ..core.functions import Threshold

from ._validate import _both_values_in_scale, _inverse_values, _all_lens_equal


def discordance_marginal(
    a_value: NumericValue,
    b_value: NumericValue,
    scale: QuantitativeScale,
    preference_threshold: Threshold,
    veto_threshold: Threshold,
    pre_veto_threshold: Optional[Threshold],
    inverse: bool = False,
) -> NumericValue:
    """_summary_
    :param NumericValue a_value: _description_
    :param NumericValue b_value: _description_
    :param QuantitativeScale scale: _description_
    :param Threshold preference_threshold: _description_
    :param Threshold veto_threshold: _description_
    :param Optional[Threshold] pre_veto_threshold: _description_
    :param bool inverse: _description_, defaults to False
    :raises ValueError: _description_
    :return NumericValue: _description_
    """
    _both_values_in_scale(a_value, b_value, scale)
    a_value, b_value, scale = _inverse_values(a_value, b_value, scale, inverse)

    try:
        veto: NumericValue = veto_threshold(a_value)
        preference: NumericValue = preference_threshold(a_value)
        pre_veto: Optional[NumericValue] = (
            pre_veto_threshold(a_value) if pre_veto_threshold is not None else None
        )
    except TypeError as exc:
        exc.args = ("",)  # TODO: Wrong threshold type
        raise

    # TODO: exception message - wrong threshold values
    if preference >= veto:
        raise ValueError("")
    if pre_veto and (pre_veto >= veto or pre_veto <= preference):
        raise ValueError("")

    if scale.preference_direction == PreferenceDirection.MAX:
        if b_value - a_value > veto:
            return 1
        elif b_value - a_value <= preference:
            return 0
        return (veto - b_value + a_value) / (veto - preference)

    if a_value - b_value > veto:
        return 1
    elif a_value - b_value <= preference:
        return 0
    return (veto - a_value + b_value) / (veto - preference)


def discordance_pair(
    a_values: List[NumericValue],
    b_values: List[NumericValue],
    scales: List[QuantitativeScale],
    preference_thresholds: List[Threshold],
    veto_thresholds: List[Threshold],
    pre_veto_thresholds: Optional[List[Threshold]],
    inverse: bool = False,
) -> List[NumericValue]:
    """_summary_
    :param List[NumericValue] a_values: _description_
    :param List[NumericValue] b_values: _description_
    :param List[QuantitativeScale] scales: _description_
    :param List[Threshold] preference_thresholds: _description_
    :param List[Threshold] veto_thresholds: _description_
    :param Optional[List[Threshold]] pre_veto_thresholds: _description_
    :param bool inverse: _description_, defaults to False
    :return List[NumericValue]: _description_
    """
    if pre_veto_thresholds:
        _all_lens_equal(
            a_values,
            b_values,
            scales,
            preference_thresholds,
            veto_thresholds,
            pre_veto_thresholds,
        )
        return [
            discordance_marginal(
                aval,
                bval,
                scale,
                preference_threshold,
                veto_threshold,
                pre_veto_threshold,
                inverse,
            )
            for aval, bval, scale, preference_threshold, veto_threshold, pre_veto_threshold in zip(
                a_values,
                b_values,
                scales,
                preference_thresholds,
                veto_thresholds,
                pre_veto_thresholds,
            )
        ]
    _all_lens_equal(a_values, b_values, scales, preference_thresholds, veto_thresholds)
    return [
        discordance_marginal(
            aval,
            bval,
            scale,
            preference_threshold,
            veto_threshold,
            inverse=inverse,
            pre_veto_threshold=None,
        )
        for aval, bval, scale, preference_threshold, veto_threshold in zip(
            a_values, b_values, scales, preference_thresholds, veto_thresholds
        )
    ]


def discordance(
    alt_values: List[NumericValue],
    performance_table: List[List[NumericValue]],
    scales: List[QuantitativeScale],
    preference_thresholds: List[Threshold],
    veto_thresholds: List[Threshold],
    pre_veto_thresholds: Optional[List[Threshold]],
) -> List[List[NumericValue]]:
    """_summary_
    :param List[NumericValue] alt_values: _description_
    :param List[List[NumericValue]] performance_table: _description_
    :param List[NumericValue] scales: _description_
    :param List[Threshold] preference_thresholds: _description_
    :param List[Threshold] veto_thresholds: _description_
    :param Optional[List[Threshold]] pre_veto_thresholds: _description_
    :return List[List[NumericValue]]: _description_
    """
    try:
        return [
            discordance_pair(
                alt_values,
                performance_table_row,
                scales,
                preference_thresholds,
                veto_thresholds,
                pre_veto_thresholds,
            )
            for performance_table_row in performance_table
        ]
    except TypeError as exc:
        exc.args = ("",)  # TODO: performance table is not a list
        raise


def counter_veto_marginal(
    a_value: NumericValue,
    b_value: NumericValue,
    scale: QuantitativeScale,
    counter_veto_threshold: Threshold,
) -> bool:
    """_summary_
    :param NumericValue a_value: _description_
    :param NumericValue b_value: _description_
    :param QuantitativeScale scale: _description_
    :param Threshold counter_veto_threshold: _description_
    :return bool: _description_
    """
    _both_values_in_scale(a_value, b_value, scale)
    try:
        counter_veto: NumericValue = counter_veto_threshold(a_value)
    except TypeError as exc:
        exc.args = ("",)  # TODO: counter_veto_threshold is not a Threshold object
        raise
    return (
        b_value - a_value > counter_veto
        if scale.preference_direction == PreferenceDirection.MAX
        else a_value - b_value > counter_veto
    )


def counter_veto_pair(
    a_values: List[NumericValue],
    b_values: List[NumericValue],
    scales: List[QuantitativeScale],
    counter_veto_thresholds: List[Threshold],
) -> List[bool]:
    """_summary_
    :param List[NumericValue] a_values: _description_
    :param List[NumericValue] b_values: _description_
    :param List[QuantitativeScale] scales: _description_
    :param List[Threshold] counter_veto_thresholds: _description_
    :return List[bool]: _description_
    """
    _all_lens_equal(a_values, b_values, scales, counter_veto_thresholds)
    return [
        counter_veto_marginal(aval, bval, scale, counter_veto_threshold)
        for aval, bval, scale, counter_veto_threshold in zip(
            a_values, b_values, scales, counter_veto_thresholds
        )
    ]


def counter_veto(
    alt_values: List[NumericValue],
    performance_table: List[List[NumericValue]],
    scales: List[QuantitativeScale],
    counter_veto_thresholds: List[Threshold],
) -> List[List[bool]]:
    """_summary_
    :param List[NumericValue] alt_values: _description_
    :param List[List[NumericValue]] performance_table: _description_
    :param List[QuantitativeScale] scales: _description_
    :param List[Threshold] counter_veto_thresholds: _description_
    :return List[List[bool]]: _description_
    """
    try:
        return [
            counter_veto_pair(
                alt_values, performance_table_row, scales, counter_veto_thresholds
            )
            for performance_table_row in performance_table
        ]
    except TypeError as exc:
        exc.args = ("",)  # TODO: preformance table is not a List
        raise


def counter_veto_count(
    performance_table: List[List[NumericValue]],
    scales: List[QuantitativeScale],
    counter_veto_thresholds: List[Threshold],
) -> List[List[int]]:
    try:
        return [
            [
                sum(
                    counter_veto_pair(
                        a_values, b_values, scales, counter_veto_thresholds
                    )
                )
                for b_values in performance_table
            ]
            for a_values in performance_table
        ]
    except TypeError as exc:
        exc.args = ("",)  # TODO performance tables is not a list
        raise