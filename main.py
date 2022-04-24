
from concordance import *
from core.scales import QuantitativeScale, PreferenceDirection

if __name__ == '__main__':
    fiat    = [100, 3500]
    vwPolo  = [150, 5000]
    scales = [QuantitativeScale(100, 1000),
              QuantitativeScale(150, 5000, PreferenceDirection.MIN)]
    weights = [2, 3]
    indifference = [
        [0.1, 10],
        [0, 500]
    ]
    preference = [
        [0.3, 20],
        [0.4, 500]
    ]
    for i in range(2):
        print(concordance_marginal(
            fiat[i], vwPolo[i], scales[i], indifference[i], preference[i], False
        ))
    for i in range(2):
        print(concordance_marginal(
            vwPolo[i], fiat[i], scales[i], indifference[i], preference[i], False
        ))

    print(concordance_comprehensive(
        fiat, vwPolo, scales, weights, indifference, preference, False
    ))
    print(concordance_comprehensive(
        vwPolo, fiat, scales, weights, indifference, preference, False
    ))
