from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Union, cast

import numpy as np

from .aliases import Function, NumericValue, Value
from .functions import FuzzyNumber, Interval


def numeric_value(value: Union[NumericValue, FuzzyNumber]) -> NumericValue:
    """Convert value to numeric.

    Used to keep numeric values as is, and convert
    :class:`functions.FuzzyNumber` to numerical value.

    :param value:
    :return:
    """
    return value.average() if isinstance(value, FuzzyNumber) else value


class PreferenceDirection(Enum):
    """Enumeration of MCDA preference directions."""

    MIN = auto()
    MAX = auto()

    @classmethod
    def has_value(cls, x: "PreferenceDirection") -> bool:
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
        return "PreferenceDirection only has following values " + s


class Scale(ABC):
    """Basic abstract class for MCDA scale."""

    @abstractmethod
    def __contains__(self, x: Value) -> bool:  # pragma: nocover
        """Check if values are inside scale.

        :param x:
        :return:
        """
        pass

    @abstractmethod
    def range(self, nb: int = None) -> List[Value]:  # pragma: nocover
        """Return range of value from scale.

        :param nb: number of values to return
        :return:
        """
        pass

    @abstractmethod
    def transform_to(
        self, x: Value, target_scale: "Scale" = None
    ) -> Value:  # pragma: nocover
        """Transform value from this scale to target scale.

        :param x:
        :param target_scale:
        :return:
        """
        pass

    def normalize(self, x: Value) -> NumericValue:
        """Normalize value.

        :param x:
        :return: normalized value
        """
        return cast(NumericValue, self.transform_to(x, get_normalized_scale()))

    def denormalize(self, x: NumericValue) -> Value:
        """Denormalize value.

        :param x: normalized value
        :return: denormalized value
        """
        return get_normalized_scale().transform_to(cast(Value, x), self)


class NominalScale(Scale):
    """This class implements a MCDA nominal scale.

    :param labels:
    """

    def __init__(self, labels: List[Value]):
        """Constructor method"""
        Scale.__init__(self)
        self.labels = labels

    def __contains__(self, x: Value) -> bool:
        """Check if values are inside scale.

        :param x:
        :return:
        """
        return x in self.labels

    def range(self, nb: int = None) -> List[Value]:
        """Return range of value from scale.

        :param nb: number of values to return (always ignored here)
        :return:
        """
        return self.labels

    def transform_to(self, x: Value, target_scale: Scale = None) -> Value:
        """Transform value from this scale to target scale.

        :param x:
        :param target_scale:
        :return:
        :raise ValueError:
            * if value `x` is outside this scale
            * if `target_scale` is not set
        :raise TypeError:
            if `target_scale` is neither :class:`QualitativeScale` nor
            :class:`NominalScale`
        """
        if x not in self:
            raise ValueError(f"label outside scale: {x}")
        if target_scale is None:
            raise ValueError("non-specified target scale")
        if isinstance(target_scale, NominalScale):
            return target_scale.labels[target_scale.labels.index(x)]
        raise TypeError("cannot transform from nominal to quantitative scale")


class QuantitativeScale(Scale, Interval):
    """Class for quantitative scale.

    :param dmin: min boundary of scale
    :param dmax: max boundary of scale
    :param preference_direction: scale preference direction
    :raises ValueError:
        * if `dmax` smaller than `dmin`
        * if `preference_direction` is unknown
    """

    def __init__(
        self,
        dmin: NumericValue,
        dmax: NumericValue,
        preference_direction: PreferenceDirection = PreferenceDirection.MAX,
    ):
        """Constructor method"""
        Interval.__init__(self, dmin, dmax)
        if not PreferenceDirection.has_value(preference_direction):
            raise ValueError(PreferenceDirection.content_message())
        self.preference_direction = preference_direction

    def __contains__(self, x: Value) -> bool:
        """Check if values are inside scale.

        :param x:
        :return:
        """
        return self.inside(cast(NumericValue, x))

    def range(self, nb: int = None) -> List[Value]:
        """Return range of value from scale.

        :param nb: number of values to return
        :return:
        """
        nb = 2 if nb is None else nb
        return cast(
            List[Value], np.linspace(self.dmin, self.dmax, nb).tolist()
        )

    def normalize_value(self, x: Value) -> NumericValue:
        """Normalize numeric value.

        :param x:
        :return:

        .. note::
            `preference_direction` is taken into account, so preferred
            value is always bigger.
        """
        if self.preference_direction == PreferenceDirection.MIN:
            return 1 - Interval.normalize(self, x)
        return Interval.normalize(self, x)

    def denormalize_value(self, x: NumericValue) -> Value:
        """Denormalize normalized numeric value.

        :param x:
        :return:

        .. note::
            `preference_direction` is taken into account, so preferred
            normalized value must always be bigger.
        """
        if self.preference_direction == PreferenceDirection.MIN:
            return Interval.denormalize(self, 1 - x)
        return Interval.denormalize(self, x)

    def transform_to(self, x: Value, target_scale: Scale = None) -> Value:
        """Transform value from this scale to target scale.

        :param x:
        :param target_scale:
        :return:
        :raise ValueError:
            * if value `x` is outside this scale
            * if `target_scale` is not set
        :raise TypeError:
            if target_scale is neither :class:`QuantitativeScale` nor
            :class:`QualitativeScale`

        .. note:: `preference_direction` attributes are taken into account
        """
        if x not in self:
            raise ValueError(f"value outside scale: {x}")
        if target_scale is None:
            raise ValueError("non-specified target scale")
        _x = cast(NumericValue, x)
        if isinstance(target_scale, QualitativeScale):
            return target_scale.label_from_value(
                target_scale.denormalize_value(self.normalize_value(_x))
            )
        if isinstance(target_scale, QuantitativeScale):
            return target_scale.denormalize_value(self.normalize_value(_x))
        raise TypeError("cannot transform from quantitative to nominal scale")

    def normalize(self, x: Value) -> NumericValue:
        """Normalize value.

        :param x:
        :return: normalized value

        .. note:: `preference_direction` attributes is taken into account
        """
        return Scale.normalize(self, x)

    def denormalize(self, x: NumericValue) -> Value:
        """Denormalize value.

        :param x: normalized value
        :return: denormalized value

        .. note:: `preference_direction` attributes is taken into account
        """
        return Scale.denormalize(self, x)


_NORMALIZED_SCALE = QuantitativeScale(0, 1)


def get_normalized_scale():
    return _NORMALIZED_SCALE


class QualitativeScale(QuantitativeScale, NominalScale):
    """This class implements a MCDA qualitative scale.

    :param labels:
    :param values:
    :param preference_direction: scale preference direction
    :raises ValueError:
        * if number of `labels` and `values` differs
        * if `preference_direction` is unknown
    :raises TypeError:
        * if `values` contains non-numeric values

    .. warning::
        This scale contains `labels` not `values`. `values` are only here to
        define a corresponding quantitative scale for default scale
        transformation. After calling :meth:`transform_to` with no associated
        scale, the data is no longer considered inside the qualitative scale.

    .. todo::
        this scale is quantitative but has a quantitative scale as an
        attribute. Find out how to best type cast it.
    """

    def __init__(
        self,
        labels: List[Value],
        values: List[NumericValue],
        preference_direction: PreferenceDirection = PreferenceDirection.MAX,
    ):
        """Constructor method"""
        if len(labels) != len(values):
            raise ValueError(
                "QualitativeScale must have same number of labels"
                " and values"
            )
        NominalScale.__init__(self, labels)
        for value in values:
            if type(value) not in [
                int,
                float,
                np.float,
                np.int,
            ]:
                raise TypeError(
                    "QualitativeScale must have numeric values."
                    f"Got: {type(value)}"
                )
        self.values = values
        QuantitativeScale.__init__(
            self,
            min(self.values),
            max(self.values),
            preference_direction,
        )
        self.__quantitative = QuantitativeScale(
            min(self.values),
            max(self.values),
            preference_direction,
        )

    @property
    def quantitative(
        self,
    ) -> QuantitativeScale:  # Property used to hide setter
        """Quantitative scale extracted from current qualitative scale."""
        return self.__quantitative

    def __contains__(self, x: Value) -> bool:
        """Check if label is inside scale.

        :param x:
        :return:
        """
        return NominalScale.__contains__(self, x)

    def range(self, nb: int = None) -> List[Value]:
        """Return range of value from scale.

        :param nb: number of values to return (always ignored here)
        :return:
        """
        return NominalScale.range(self, nb)

    def transform_to(self, x: Value, target_scale: Scale = None) -> Value:
        """Transform value from this scale to target scale.

        :param x:
        :param target_scale:
        :return:
        :raise ValueError: if value `x` is outside this scale
        :raise TypeError: if `target_scale` has unknown type

        .. note::
            `preference_direction` attributes are taken into account when
            rescaling to a :class:`QuantitativeScale`
        """
        if x not in self:
            raise ValueError(f"label outside scale: {x}")
        target_scale = (
            self.quantitative if target_scale is None else target_scale
        )
        if isinstance(target_scale, NominalScale):
            return target_scale.labels[target_scale.labels.index(x)]
        if isinstance(target_scale, QuantitativeScale):
            value = self.values[self.labels.index(x)]
            return target_scale.denormalize_value(self.normalize_value(value))
        raise TypeError(
            f"unrecognized scale type for scale: {target_scale}"
        )  # pragma: nocover

    def label_from_value(self, x: Value) -> Value:
        """Transform value to this scale.

        :param x:
        :raises ValueError: if `x` corresponds to no label
        :return: label associated to given value
        """
        if x not in self.values:
            raise ValueError(f"value outside scale: {x}")
        _x = cast(NumericValue, x)
        return self.labels[self.values.index(_x)]


class FuzzyScale(QualitativeScale):
    """This class implements a MCDA fuzzy qualitative scale.

    :param labels:
    :param fuzzy:
    :param preference_direction: scale preference direction
    :raises ValueError:
        * if number of `labels` and `fuzzy` differs
        * if `preference_direction` is unknown
    :raises TypeError:
        * if `fuzzy` contains non-fuzzy numbers
    """

    def __init__(
        self,
        labels: List[Value],
        fuzzy: List[FuzzyNumber],
        preference_direction: PreferenceDirection = PreferenceDirection.MAX,
    ):
        values = []
        for fz in fuzzy:
            if type(fz) is not FuzzyNumber:
                raise TypeError("fuzzy scales can only contains fuzzy numbers")
            values.append(fz.average())
        QualitativeScale.__init__(self, labels, values, preference_direction)
        self.fuzzy = fuzzy


class ScaledFunction:
    """This class implements functions with an input definition domain.

    The input definition domain is represented by a :class:`Scale` object.

    :param function:
    :param scale_input:
    """

    def __init__(self, function: Function, scale_input: Scale):
        self.function = function
        self.scale_input = scale_input

    def transform_to(self, scale: Scale) -> "ScaledFunction":
        """Return same function with a transformed definition domain.

        Transformation from input `scale` to attribute `scale_input` is
        composed to attribute `function` in order to create the new
        function.

        :param scale:
        :return:
        """
        return ScaledFunction(
            lambda x: self.function(scale.transform_to(x, self.scale_input)),
            scale,
        )

    def __call__(self, x: Value) -> Value:
        """Call function.

        :param x:
        :return:
        :raise ValueError: if `x` outside of `scale_input`
        """
        if x not in self.scale_input:
            raise ValueError(f"input is outside its scale: {x}")
        return self.function(x)
