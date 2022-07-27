from core.aliases import NumericValue
from typing import List


def get_relation_type(x, y, outranking):
    '''
    Assigns type of relation according to credibility
    :param x:
    :param y:
    :param outranking:
    :return:
    '''
    if outranking[x][y] and outranking[y][x]:
        relation = 'indifference'
    elif outranking[x][y] and not outranking[y][x]:
        relation = 'preference'
    elif not outranking[x][y] and not outranking[y][x]:
        relation = 'incomparability'
    else:
        relation = None
    return relation


def assign_TriC_class(alternatives: List[NumericValue],
                      categories_rank: dict,
                      categories_profiles: dict,
                      outranking: dict,
                      credibility: dict):

    # sorted categories by ranks - ascending (worst to best)
    categories = [i[0] for i in sorted(categories_rank.items(), key=lambda x: x[1], reverse=True)]
    # list of profiles according to categories
    profiles = [i[0] for i in sorted(categories_profiles.items(), key=lambda x: categories.index(x[1]))]
    assignments_descending = []
    assignments_ascending = []
    for a in alternatives:
        found_descending = False
        for p in profiles[len(profiles) - 2:: -1]:
            p_next = profiles[profiles.index(p) + 1]
            relation = get_relation_type(a, p, outranking)
            relation_next = get_relation_type(a, p_next, outranking)
            if (relation == 'preference' and
                    (credibility[a][p_next] > credibility[p][a] or
                     credibility[a][p_next] >= credibility[p][a] and
                     relation_next == 'incomparability')):
                category = categories_profiles.get(p_next)
                assignments_descending.append((a, category))
                found_descending = True
                break
        if not found_descending:
            assignments_descending.append((a, categories[0]))
            
        found_ascending = False
        for p in profiles[1:]:
            p_prev = profiles[profiles.index(p) - 1]
            relation = get_relation_type(p, a, outranking)
            relation_prev = get_relation_type(a, p_prev, outranking)
            if (relation == 'preference' and
                    (credibility[p_prev][a] > credibility[a][p] or
                     credibility[p_prev][a] >= credibility[a][p] and
                     relation_prev == 'incomparability')):
                category = categories_profiles.get(p_prev)
                assignments_ascending.append((a, category))
                found_ascending = True
                break
        if not found_ascending:
            assignments_ascending.append((a, categories[-1]))
    assignments = {}
    for i in zip(assignments_descending, assignments_ascending):
        assignments[i[0][0]] = (i[0][1], i[1][1])
    return assignments


def assign_TriB_class(alternatives: List,
                      categories_rank: dict,
                      categories_profiles: dict,
                      crisp_outranking: dict):
    """

    :param alternatives:
    :param categories_rank:
    :param categories_profiles:
    :param crisp_outranking:
    :return:
    """
    # Initiate categories to assign
    categories = [i[0] for i in sorted(categories_rank.items(), key=lambda x: x[1], reverse=True)]
    assignment = {}
    for alternative in alternatives:
        # Pessimistic assignment
        pessimistic_idx = 0
        for i, profile in list(enumerate(categories_profiles))[::-1]:
            relation = get_relation_type(alternative, profile, crisp_outranking)
            if relation in ('indifference', 'preference'):
                pessimistic_idx = i + 1
                break

        # Optimistic assignment
        optimistic_idx = len(categories_profiles)
        for i, profile in enumerate(categories_profiles):
            relation = get_relation_type(profile, alternative, crisp_outranking)
            if relation == 'preference':
                optimistic_idx = i
                break

        assignment[alternative] = (categories[pessimistic_idx], categories[optimistic_idx])
    return assignment
