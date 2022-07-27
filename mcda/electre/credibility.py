"""This module implements methods to compute
an outranking credibility."""

from typing import List, Union

from ..core.aliases import NumericValue


def credibility_cv_pair(
    concordance_comprehensive: NumericValue,
    discordance_comprehensive: NumericValue,
    counter_veto_occurs: List[Union[int, bool]],
) -> NumericValue:
    """_summary_

    :param NumericValue concordance_comprehensive: _description_
    :param NumericValue discordance_comprehensive: _description_
    :param List[Union[int, bool]] counter_veto_occurs: _description_

    :return NumericValue: _description_
    """
    try:
        return concordance_comprehensive * discordance_comprehensive ** (
            1 - sum(counter_veto_occurs) / len(counter_veto_occurs)
        )
    except TypeError as exc:
        exc.args = ("",)
        raise  # TODO


def credibility_cv(
    concordance_comprehensive: List[List[NumericValue]],
    discordance_comprehensive: List[List[NumericValue]],
    counter_veto_occurs: List[List[int]],
) -> List[List[NumericValue]]:
    """_summary_

    :param List[List[NumericValue]] concordance_comprehensive: _description_
    :param List[List[NumericValue]] discordance_comprehensive: _description_
    :param List[List[int]] counter_veto_occurs: _description_

    :return List[List[NumericValue]]: _description_
    """
    try:
        return [
            [
                credibility_cv_pair(concordance, discordance, [cv])
                for concordance, discordance, cv in zip(
                    concordance_row, discordance_row, cv_row
                )
            ]
            for concordance_row, discordance_row, cv_row in zip(
                concordance_comprehensive,
                discordance_comprehensive,
                counter_veto_occurs,
            )
        ]
    except TypeError as exc:
        exc.args = ("",)
        raise  # TODO
