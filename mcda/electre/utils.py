from mcda.core.aliases import NumericValue


def linear_function(
    alpha: NumericValue, x: NumericValue, beta: NumericValue
) -> NumericValue:
    """Calculates linear function.

    :param alpha: coefficient of the independent variable
    :param x: independent variable
    :param beta: y-intercept
    :return: dependent variable
    """
    return alpha * x + beta
