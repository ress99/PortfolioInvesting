import random

def as_mutate(ind):

    tic_to_insert = random.sample(ind.all_assets, 1)[0]
    tic_to_remove = random.sample(ind.asset_list, 1)[0]

    ind.substitute_assets(tic_to_remove, tic_to_insert)

    return ind

def po_mutate(ind):

    tic1 = random.sample(ind.asset_list, 1)[0]
    tic2 = random.sample(ind.asset_list, 1)[0]
    tic1_weight = ind.prtf_dict[tic1]['weight']
    tic2_weight = ind.prtf_dict[tic2]['weight']

    ind.change_asset_weight(tic1, tic2_weight)
    ind.change_asset_weight(tic2, tic1_weight)

    return ind