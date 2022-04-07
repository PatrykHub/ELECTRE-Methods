#pragma once
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
        print(concordanceMarginal(
            fiat[i], vwPolo[i], scales[i], indifference[i], preference[i]
        ))
    for i in range(2):
        print(concordanceMarginal(
            vwPolo[i], fiat[i], scales[i], indifference[i], preference[i]
        ))

    print(concordanceComprehensive(
        fiat, vwPolo, scales, weights, indifference, preference
    ))
    print(concordanceComprehensive(
        vwPolo, fiat, scales, weights, indifference, preference
    ))

    interactions = [[[], []],
                    [[], []]]
    interact(interactions, 1, 1, 'MS', 'min', 4)
    print(interactions)

    print(concordanceWithInteractions(
        vwPolo, fiat, scales, weights, interactions, indifference, preference
    ))