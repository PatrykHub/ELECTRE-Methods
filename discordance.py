
from core.scales import QuantitativeScale, PreferenceDirection
from core.aliases import NumericValue
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

