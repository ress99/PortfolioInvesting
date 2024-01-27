import random

import data_op as op

# def aux_get_swap_numbers(lenght):

#     nmbr1 = random.randint(0, lenght)
#     nmbr2 = random.randint(0, lenght)

#     if nmbr1 >= nmbr2:
#         nmbr1, nmbr2 = aux_get_swap_numbers(lenght)

#     return nmbr1, nmbr2

def as_mutate(ind):

    tic_to_insert = random.sample(ind.all_assets, 1)[0]
    tic_to_remove = random.sample(ind.asset_list, 1)[0]

    ind.swap_assets(tic_to_remove, tic_to_insert)

    return ind

def as_mutate_all(ind, prob = 0.4):

    for tic_to_remove in ind.asset_list:
        if random.random() < prob:
            tic_to_insert = random.sample(ind.all_assets, 1)[0]
            ind.swap_assets(tic_to_remove, tic_to_insert)

    return ind

def po_reverse_weights(ind):

    weights = ind.asset_weights
    nmbr1, nmbr2 = op.aux_get_swap_numbers(len(weights))

    weights[nmbr1:nmbr2+1] = reversed(weights[nmbr1:nmbr2+1])

    ind.asset_weights = weights

    return ind


def po_swap_weights(ind):

    tic1 = random.sample(ind.asset_list, 1)[0]
    tic2 = random.sample(ind.asset_list, 1)[0]
    tic1_weight = ind.prtf_dict[tic1]['weight']
    tic2_weight = ind.prtf_dict[tic2]['weight']

    ind.change_asset_weight(tic1, tic2_weight)
    ind.change_asset_weight(tic2, tic1_weight)

    return ind