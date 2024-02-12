import random

import data_op as op

def as_mate(ind1, ind2):
    tic1 = random.sample(ind1.asset_list, 1)[0]
    tic2 = random.sample(ind2.asset_list, 1)[0]

    ind1.swap_assets(tic1, tic2)
    ind2.swap_assets(tic2, tic1)

    return ind1, ind2


def po_mate(ind1, ind2):

    length = len(ind1.asset_weights)
    nmbr1, nmbr2 = op.aux_get_swap_numbers(length)

    weights1 = ind1.asset_weights
    weights2 = ind2.asset_weights

    swap1 = weights1[nmbr1:nmbr2]
    swap2 = weights2[nmbr1:nmbr2]
    weights1[nmbr1:nmbr2] = swap2
    weights2[nmbr1:nmbr2] = swap1

    ind1.asset_weights = weights1
    ind2.asset_weights = weights2

    return ind1, ind2