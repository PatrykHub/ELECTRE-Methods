class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

def get_credibility(comparables_a, comparables_b, concordance, discordance,
                    with_denominator, only_max_discordance, use_partials):

    def _get_credibility_index(x, y, with_denominator, only_max_discordance,
                               use_partials):
        if use_partials:
            discordance_values = discordance[x][y].values()
        else:
            discordance_values = [discordance[x][y]]
        if set(discordance_values) == set([0]):  # only zeros
            c_idx = concordance[x][y]
        elif 1 in discordance_values:            # at least one '1'
            if not concordance[x][y] < 1:
                raise RuntimeError("When discordance == 1, "
                                   "concordance must be < 1.")
            c_idx = 0.0
        elif only_max_discordance and not with_denominator:
            c_idx = concordance[x][y] * (1 - max(discordance_values))
        else:
            factors = []
            for d in discordance_values:
                factor = None
                if with_denominator:
                    if d > concordance[x][y]:
                        factor = (1 - d) / (1 - concordance[x][y])
                else:
                    factor = (1 - d)
                if factor:
                    factors.append(factor)
            if factors == []:
                c_idx = concordance[x][y]
            else:
                c_idx = concordance[x][y] * reduce(lambda f1, f2: f1 * f2,
                                                   factors)
        return c_idx

    two_way_comparison = True if comparables_a != comparables_b else False
    credibility = Vividict()
    for a in comparables_a:
        for b in comparables_b:
            credibility[a][b] = _get_credibility_index(a, b, with_denominator,
                                                       only_max_discordance,
                                                       use_partials)
            if two_way_comparison:
                credibility[b][a] = _get_credibility_index(b, a, with_denominator,
                                                           only_max_discordance,
                                                           use_partials)
    return credibility