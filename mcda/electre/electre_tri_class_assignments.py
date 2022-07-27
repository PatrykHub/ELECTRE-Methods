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


def assign_TriB_class(alternatives: List,
                 categories_rank: dict,
                 categories_profiles: List,
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
