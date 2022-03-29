import math
from typing import Any, Dict, List, Optional, cast

from .aliases import NumericFunction, NumericValue, Value


def affine_function_from_segment(
    segment: List[List[NumericValue]],
) -> NumericFunction:
    """Creates an affine function from segment.

    Each segment of `segments` is a list of two points, each point a
    sequence of numeric abscissa, a numeric ordinate and a :class:`bool`
    indicating if the point is included in the underlying interval  ;

    :param segment:
    :return: affine function
    """
    if len(segment) < 2:
        raise ValueError("two points are needed to define an affine function")
    if len(segment[0]) < 2 or len(segment[1]) < 2:
        raise ValueError(
            "two coordinates per point are needed to "
            "define an affine function"
        )
    if segment[0][0] == segment[1][0]:
        raise ValueError(
            "affine function needs two points with different abscissa"
        )
    slope = (segment[0][1] - segment[1][1]) / (segment[0][0] - segment[1][0])
    constant = (
        segment[1][1] * segment[0][0] - segment[0][1] * segment[1][0]
    ) / (segment[0][0] - segment[1][0])
    return lambda x: slope * x + constant


class Interval:
    """This class describes a numeric interval.

    :param dmin: min boundary of interval
    :param dmax: max boundary of interval
    :param min_in: is min boundary inside interval or not
    :param max_in: is max boundary inside interval or not
    :raises ValueError: if `dmin` bigger than or equal to `dmax`

    .. note :: Empty or simplex intervals are forbidden by construction
    """

    def __init__(
        self,
        dmin: NumericValue,
        dmax: NumericValue,
        min_in: bool = True,
        max_in: bool = True,
    ):
        """Constructor method"""
        if dmin >= dmax:
            raise ValueError(
                f"Interval min value '{dmin}' bigger than or "
                f"equal to max value '{dmax}'"
            )
        self.dmin = dmin
        self.dmax = dmax
        self.min_in = min_in
        self.max_in = max_in

    def __contains__(self, x: NumericValue) -> bool:
        """Check whether value is inside interval or not.

        :param x:
        :return:
        """
        return self.inside(x)

    def inside(self, x: NumericValue) -> bool:
        """Check whether value is inside interval or not.

        :param x:
        :return:
        :rtype: bool
        """
        if self.dmin < x < self.dmax:
            return True
        if self.min_in and x == self.dmin:
            return True
        if self.max_in and x == self.dmax:
            return True
        return False

    def normalize(self, x: Value) -> NumericValue:
        """Normalize value inside interval.

        :param x:
        :return:
        """
        _x = cast(NumericValue, x)
        return (
            (_x - self.dmin) / (self.dmax - self.dmin)
            if self.dmin != self.dmax
            else 0
        )

    def denormalize(self, x: NumericValue) -> Value:
        """Denormalize normalized value inside interval.

        :param x:
        :return:
        """
        return cast(Value, x * (self.dmax - self.dmin) + self.dmin)

    def join(self, other: "Interval") -> "Interval":
        """Compute maximal junction between two intervals.

        Biggest interval containing both intervals.

        :param other:
        :return:
        """
        dmin = min((self.dmin, other.dmin))
        dmax = max((self.dmax, other.dmax))
        min_in = (self.min_in if dmin == self.dmin else False) or (
            other.min_in if dmin == other.dmin else False
        )
        max_in = (self.max_in if dmax == self.dmax else False) or (
            other.max_in if dmax == other.dmax else False
        )
        return Interval(dmin, dmax, min_in, max_in)

    def intersect(self, other: "Interval") -> Optional["Interval"]:
        """Compute intersection between two intervals.

        :param other:
        :type other: Interval
        :return:
        :rtype: Interval or None
        """
        dmin = max((self.dmin, other.dmin))
        dmax = min((self.dmax, other.dmax))
        min_in = (self.min_in if dmin == self.dmin else True) and (
            other.min_in if dmin == other.dmin else True
        )
        max_in = (self.max_in if dmax == self.dmax else True) and (
            other.max_in if dmax == other.dmax else True
        )
        try:
            res = Interval(dmin, dmax, min_in, max_in)
            return res
        except ValueError:
            return None

    def union(self, other: "Interval") -> Optional["Interval"]:
        """Compute union of two intervals.

        :param other:
        :return:

        .. note :: Returns ``None`` if intervals don't coïncide
        """
        dmin = max((self.dmin, other.dmin))
        dmax = min((self.dmax, other.dmax))
        min_in = (self.min_in if dmin == self.dmin else True) and (
            other.min_in if dmin == other.dmin else True
        )
        max_in = (self.max_in if dmax == self.dmax else True) and (
            other.max_in if dmax == other.dmax else True
        )
        if dmin > dmax:
            return None
        if dmin == dmax:
            if not min_in or not max_in:
                return None
        return self.join(other)

    def continuous(self, other: "Interval") -> bool:
        """Check continuity with following interval.

        :param other:
        :return:

        .. note ::
            Strict continuity is checked (i.e if the intervals touches without
            overlapping).
            In other words : `dmax` equal to `other`'s `dmin`
        """
        if self.dmax != other.dmin:
            return False
        return self.max_in or other.min_in

    def __eq__(self, other: Any) -> bool:
        """Checks both intervals share the same fields.

        :param other:
        :return:
        """
        return (
            self.dmin == other.dmin
            and self.dmax == other.dmax
            and self.min_in == other.min_in
            and self.max_in == other.max_in
        )


class DiscreteFunction:
    """This class implements discrete function.

    :param values: function description, abscissa as keys, ordinates as values
    """

    def __init__(self, values: Dict[Value, Value]):
        """Constructor method"""
        self.values = {}
        self.values.update(values)

    def apply(self, x: Value) -> Value:
        """Apply function to single value.

        :param x:
        :return:
        :raises IndexError: if `x` is not in `values`
        """
        if x not in self.values:
            raise IndexError(f"discrete value '{x}' unknown")
        return self.values[x]

    def __call__(self, x: Value) -> Value:
        """Apply function to single value.

        :param x:
        :return:
        :raises IndexError: if `x` is not in `values`
        """
        return self.apply(x)


class PieceWiseFunction:
    """This class implements piecewise MCDA function.

    :param intervals: functions definition intervals
    :param functions:
    :param segments: list of segments defining piecewise linear functions
    :raises ValueError: if number of intervals and functions are different

    .. note::
        * first matching interval is used to return result
        * each segment of `segments` is a list of two points, each point a
          sequence of numeric abscissa, a numeric ordinate and a :class:`bool`
          indicating if the point is included in the underlying interval
    """

    _TYPE = "PieceWiseFunction"

    def __init__(
        self,
        intervals: List[Interval] = None,
        functions: List[NumericFunction] = None,
        segments: List[List[List]] = None,
    ):
        """Constructor method"""
        intervals = [] if intervals is None else intervals
        functions = [] if functions is None else functions
        segments = [] if segments is None else segments
        if len(intervals) != len(functions):
            raise ValueError(
                f"{self._TYPE} must have as many intervals as functions"
            )
        self.intervals = intervals
        self.functions = functions
        self._parse_segments(segments=segments)

    def _parse_segments(self, segments: List[List[List]]):
        """Parse segments and populate intervals and functions.

        Each segment of `segments` is a list of two points, each point a
        sequence of numeric abscissa, a numeric ordinate and a :class:`bool`
        indicating if the point is included in the underlying interval.

        :param segments:
        """
        for seg in segments:
            a = seg[0] if len(seg[0]) > 2 else seg[0] + [True]
            b = seg[1] if len(seg[1]) > 2 else seg[1] + [True]
            self.intervals.append(Interval(a[0], b[0], a[2], b[2]))
            self.functions.append(affine_function_from_segment(seg))

    def continuous(self) -> bool:
        """Check intervals and functions are ordered and continuous.

        :return:
        """
        if len(self.intervals) <= 1:
            return True
        for i in range(len(self.intervals) - 1):
            if not self.intervals[i].continuous(self.intervals[i + 1]):
                return False
            if not math.isclose(
                self.functions[i](self.intervals[i].dmax),
                self.functions[i + 1](self.intervals[i + 1].dmin),
            ):
                return False
        return True

    def apply(self, x: NumericValue) -> NumericValue:
        """Apply function to single value.

        :param x:
        :raises ValueError: if `x` is not inside an interval
        :return:
        """
        for interval, f in zip(self.intervals, self.functions):
            if x in interval:
                return f(x)
        raise ValueError(
            f"cannot apply piecewise function to out-of-bound value: {x}"
        )

    def __call__(self, x: NumericValue) -> NumericValue:
        """Apply function to single value.

        :param x:
        :raises ValueError: if `x` is not inside an interval
        :return:
        """
        return self.apply(x)


class FuzzyNumber(PieceWiseFunction):
    """This class implements a fuzzy number.

    :param intervals: functions definition intervals
    :param functions:
    :param segments: list of segments defining piecewise linear functions
    :raises ValueError:
        * if number of intervals and functions are different
        * if number of intervals and functions is zero
        * if functions are discontinuous
    """

    def __init__(
        self,
        intervals: List[Interval] = None,
        functions: List[NumericFunction] = None,
        segments: List[List[List]] = None,
    ):
        """Constructor method"""
        PieceWiseFunction.__init__(self, intervals, functions, segments)
        if len(self.functions) == 0:
            raise ValueError("FuzzyNumber must have at least one function")
        if not self.continuous():
            raise ValueError("FuzzyNumber functions must be continuous")

    def average(self) -> NumericValue:
        """Computes the average of all intervals boundaries.

        :return:
        """
        res = self.intervals[0].dmin
        for interval in self.intervals:
            res += interval.dmax
        return res / (len(self.functions) + 1)
