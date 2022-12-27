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
    """Raised, if reinforcement factor has improper
    value (is not greater than one)."""


class InconsistentCriteriaNamesError(ValueError):
    """Raised, if criteria names inside dictionaries or
    series are inconsistent, i.e. contain different values set."""


class NotUniqueNamesError(ValueError):
    """Raised, if column / index names are not unique."""


class WrongIndexValueError(ValueError):
    """Raised, if the provided index value is outside
    its permissible interval."""


class WrongInteractionTypeError(ValueError):
    """Raised, if the interaction type is not an enum type."""
