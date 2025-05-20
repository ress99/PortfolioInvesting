"""Module with implementation of mating functions for the genetic algorithm."""

import random

import data_op as op


def as_mate(ind1, ind2):
    """
    Implementation of mating function for the Asset Selection module.

    Performs a mating operation by swapping one random asset between two individuals.
    It modifies the asset lists of the individuals in place.

    Args:
        ind1: The first individual object
        ind2: The second individual object

    Returns:
        tuple: A tuple containing the two modified individuals.
    """

    #Select one random asset from each individual
    asset1 = random.sample(ind1.asset_list, 1)[0]
    asset2 = random.sample(ind2.asset_list, 1)[0]

    #Swap the assets between the two individuals
    ind1.swap_assets(asset1, asset2)
    ind2.swap_assets(asset2, asset1)

    return ind1, ind2


def po_mate(ind1, ind2):
    """
    Implementation of mating function for the Portfolio Optimization module.

    This function performs a mating operation by swapping a segment of asset weights 
    between two individuals. The segment is defined by two random indices, and the 
    weights are exchanged in place.

    Args:
        ind1: The first individual object
        ind2: The second individual object

    Returns:
        tuple: A tuple containing the two modified individuals.
    """

    #Select two random numbers to use as index in the operation
    nmbr1, nmbr2 = op.aux_get_swap_numbers(len(ind1.asset_weights))

    #Gets weights of assets of the two individuals
    weights1 = ind1.asset_weights
    weights2 = ind2.asset_weights

    #Gets weights of assets between both indexes
    swap1 = weights1[nmbr1:nmbr2]
    swap2 = weights2[nmbr1:nmbr2]

    #Swaps the weights between the two individuals
    weights1[nmbr1:nmbr2] = swap2
    weights2[nmbr1:nmbr2] = swap1

    #Sets the new weights to the individuals
    ind1.asset_weights = weights1
    ind2.asset_weights = weights2

    #Returns the two individuals with the new weights
    return ind1, ind2
