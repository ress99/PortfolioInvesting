"""Module with implementation of mutation functions for the genetic algorithm."""

import random
import data_op as op


def as_mutate(ind):
    """
    Implementation of mutation function for the Asset Selection module.

    This function performs a mutation operation by replacing one random asset in the 
    individual's asset list with another random asset from the list of all available assets. 
    The mutation modifies the individual's asset list in place.

    Args:
        ind: The individual object to be mutated.

    Returns:
        The modified individual after the mutation operation.
    """

    # Select one random asset from the list of all assets
    asset_to_insert = random.sample(ind.all_assets, 1)[0]
    # Select one random asset to remove from the individual's asset list
    asset_to_remove = random.sample(ind.asset_list, 1)[0]

    # Swap the selected assets
    ind.swap_assets(asset_to_remove, asset_to_insert)

    #Return the modified individual
    return ind


def as_mutate_all(ind, prob = 0.4):
    """
    Implementation of mutation function for the Asset Selection module.

    This function performs a mutation operation on all assets in the individual's asset list. 
    Each asset has a probability of being replaced by a random asset from the list of all 
    available assets. The mutation modifies the individual's asset list in place.

    Args:
        ind: The individual object to be mutated.
        prob (float): The probability of mutating each asset in the individual's asset list.

    Returns:
        The modified individual after the mutation operation.
    """

    #Runs through all assets of the individual
    for asset_to_remove in ind.asset_list:
        #Uses a random number to determine if the asset will be mutated
        if random.random() < prob:
            #Selects a random asset from the list of all assets
            tic_to_insert = random.sample(ind.all_assets, 1)[0]
            #Swaps the selected assets
            ind.swap_assets(asset_to_remove, tic_to_insert)

    #Returns the modified individual
    return ind


def po_reverse_weights(ind):
    """
    Implementation of mutation function for the Portfolio Optimization module.

    This function performs a mutation operation by reversing the asset weights between
    two randomly selected indices in the individual's asset weights list. 

    Args:
        ind: The individual object to be mutated.

    Returns:
        The modified individual after the mutation operation.
    """

    # Select two random numbers to use as index in the operation
    weights = ind.asset_weights
    idx1, idx2 = op.aux_get_swap_numbers(len(weights))

    #Reverse weight distribution between the two indexes
    weights[idx1:idx2+1] = reversed(weights[idx1:idx2+1])

    #Set the new weights to the individual
    ind.asset_weights = weights

    #Return the modified individual
    return ind


def po_swap_weights(ind):
    """
    Implementation of mutation function for the Portfolio Optimization module.

    This function performs a mutation operation by swapping the weights of two randomly 
    selected assets in the individual's portfolio.

    Args:
        ind: The individual object to be mutated.

    Returns:
        The modified individual after the mutation operation.
    """

    # Select two random assets to swap weights
    asset1 = random.sample(ind.asset_list, 1)[0]
    asset2 = random.sample(ind.asset_list, 1)[0]

    # Get the weights of the selected assets
    asset1_weight = ind.prtf_dict[asset1]['weight']
    asset2_weight = ind.prtf_dict[asset2]['weight']

    # Swap the weights between the two assets
    ind.change_asset_weight(asset1, asset1_weight)
    ind.change_asset_weight(asset2, asset2_weight)

    # Return the modified individual
    return ind
