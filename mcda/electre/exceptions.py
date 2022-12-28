"""This module implements some exception classes,
uses for raising errors during the validation process,
just to make debugging a little bit simpler.

Most exception classes are derived from `ValueError` class.
"""


class WrongThresholdValueError(ValueError):
    """Raised, if threshold value is improper
    for further calculations."""


class ValueOutsideScaleError(ValueError):
    """Raised, if criterion value is outside its scale."""


class WrongWeightValueError(ValueError):
    """Raised, if weight value is improper."""


class WrongFactorValueError(ValueError):
    """Raised, if reinforcement or interaction factor
    has improper value."""


class InconsistentCriteriaNamesError(ValueError):
    """Raised, if criteria names inside dictionaries or
    series are inconsistent, i.e. contain different values set."""


class InconsistentDataFrameIndexingError(ValueError):
    """Raised, if index/column values set is not the same for
    at least two different data frames."""


class NotUniqueNamesError(ValueError):
    """Raised, if column / index names are not unique."""


class WrongIndexValueError(ValueError):
    """Raised, if the provided index value is outside
    its permissible interval (or is simply missing)."""


class WrongInteractionError(ValueError):
    """Raised, if the interaction type is not an enum type
    or when criterion is interact with itself."""


class PositiveNetBalanceError(ValueError):
    """Raised, if the positive net balance condition is not fulfilled
    during the concordance with interactions calculations."""
