from typing import Callable, List, NamedTuple, Union

NumericValue = Union[int, float]
Value = Union[NumericValue, str]
Id = str
Name = str
Active = bool
Real = bool

Alternative = Id
Criterion = Id
Category = Id

PerformanceTable = List[List[Value]]
NumericPerformanceTable = List[List[NumericValue]]

Function = Callable[[Value], Value]
NumericFunction = Callable[[NumericValue], NumericValue]


class CategoriesInterval(NamedTuple):
    lower_bound: Category
    upper_bound: Category


Assignment = Union[Category, List[Category], CategoriesInterval]

# Threshold = AffineFunction