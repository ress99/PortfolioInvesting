"""Portfolio Constraints Module"""

# pylint: disable=arguments-differ

# import random
# from functools import partial
from random import choice
# import numpy as np
# import pandas as pd
# import time
from enum import Enum
from collections import Counter


class ConstraintOption(Enum):
    """Class to define the 3 options for portfolio constraints."""
    PENALIZE = 1
    INVALIDATE = 2
    ADJUST = 3


class Constraint:
    """
    Abstract base class for portfolio constraints.

    Subclasses should implement the check_constraint and apply_constraint methods
    to define specific constraint logic for portfolios.
    """


    def check_constraint(self, *args, **kwargs):
        """
        Checks whether the constraint is satisfied.

        Should be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclasses must implement the 'check_constraint' method.")


    def apply_constraint(self, *args, **kwargs):
        """
        Applies the constraint to the portfolio or individual.

        Should be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclasses must implement the 'apply_constraint' method.")


    def penalize_individual(self, portfolio, penalty = 0.1):

        # Penalize the portfolio by adding a penalty to its fitness value
        portfolio.penalty += penalty
        return


    def invalidate_individual(self, portfolio):

        # Invalidate the portfolio by setting its invalid flag to True
        portfolio.invalid = True
        return


class MinimumWeight(Constraint):

    def __init__(self, threshold = 0.05):

        self.threshold = threshold

    def check_constraint(self, portfolio):

        # Check if all asset weights are above the threshold
        weights = portfolio.asset_weights
        if min(weights) < self.threshold:
            return False
        return True


    def apply_constraint(self, portfolio, option = ConstraintOption.ADJUST):

        # Check if the constraint is satisfied
        if self.check_constraint(portfolio):
            return True

        # If the constraint is not satisfied, apply the constraint option
        if option == ConstraintOption.PENALIZE:
            self.penalize_individual(portfolio)
        elif option == ConstraintOption.INVALIDATE:
            self.invalidate_individual(portfolio)
        elif option == ConstraintOption.ADJUST:
            self.adjust_individual(portfolio)

        return


    def adjust_individual(self, portfolio):

        # Delete previous fitness values before adjusting weights
        del portfolio.fitness.values

        # Adjust the weights of the portfolio to ensure all weights are above the threshold
        adjusted_weights = portfolio.asset_weights.copy()
        # Calculate the total deficit of weights below the threshold
        deficit = sum([self.threshold - w for w in adjusted_weights if self.threshold > w])

        # Create a sorted tuple (index, weight) from which to cover the deficit 
        surplus_assets = sorted([(i, w) for i, w in enumerate(adjusted_weights)], key=lambda x: x[1], reverse=True)

        #Calculate differences between adjacent weights to determine how much to take from each asset
        differences = [surplus_assets[i][1] - surplus_assets[i + 1][1] for i in range(len(surplus_assets) - 1)]
        to_take_list = []

        # Adjust the weights of the surplus assets to cover the deficit
        adjusted_weights = [max(i, self.threshold) for i in adjusted_weights]


        for i, diff in enumerate(differences):
            if ((i + 1) * diff) > deficit:
                to_take = deficit / (i + 1)
            else:
                to_take = diff
            deficit -= to_take * (i + 1)
            to_take_list.append(0)
            to_take_list = [x + to_take for x in to_take_list]
            if deficit < 0.000005:
                break

        # for i in range(len(differences)):
        #     if ((i + 1) * differences[i]) > deficit:
        #         to_take = deficit / (i + 1)
        #     else:
        #         to_take = differences[i]
        #     deficit -= to_take * (i + 1)
        #     to_take_list.append(0)
        #     to_take_list = [i + to_take for i in to_take_list]
        #     if deficit < 0.000005:
        #         break

        # Adjust the weights of the surplus assets to cover the deficit
        for idx, value in enumerate(to_take_list):
            adjusted_weights[surplus_assets[idx][0]] -= value

        #Update the portfolio with the adjusted weights
        portfolio.asset_weights = adjusted_weights

        return


class MaxAssetsIndex(Constraint):

    def __init__(self, index_list, nmbr_assets_list):

        # Validate the inputs; raise exceptions if invalid
        self.validate_inputs(index_list, nmbr_assets_list)

        #Store dictionary with max #assets for each index
        self.index_list = index_list
        self.nmbr_assets_list = nmbr_assets_list
        inside_dict = [{'object': obj, 'number': num}
                        for _, (obj, num)
                        in enumerate(zip(index_list, nmbr_assets_list))]
        self.dict = dict(zip([i.name for i in index_list], inside_dict))


    def validate_inputs(self, index_list, nmbr_assets_list):
        """
        Validates the inputs for the MaxAssetsIndex constraint.

        Args:
            index_list (list): List of index objects.
            nmbr_assets_list (list): List of numbers of assets for each index.

        Returns:
            bool: True if inputs are valid, False otherwise.

        Raises:
            TypeError: If either input is not a list.
            ValueError: If nmbr_assets_list contains non-integer or zero/negative values.
        """

        if not isinstance(index_list, list):
            raise TypeError("index_list must be a list.")
        if not isinstance(nmbr_assets_list, list):
            raise TypeError("nmbr_assets_list must be a list.")
        if not all(isinstance(n, int) and n >= 0 for n in nmbr_assets_list):
            raise ValueError("nmbr_assets_list must be a list of non-zero positive integers.")
        if len(index_list) != len(nmbr_assets_list):
            raise ValueError("index_list and nmbr_assets_list must have the same length.")
        return True


    def check_constraint(self, portfolio):

        # If #assets in the portfolio is <= #assets in the constraint, return
        if sum(self.nmbr_assets_list) > len(portfolio):
            return False

        # Get counter of assets in the portfolio for each index
        count = self.get_count(portfolio)
        # Check if the count of each index is >= to the number specified in the constraint
        return all(count[index] >= self.dict[index]['number'] for index in self.dict)

        # for index in self.dict:
        #     if count[index] >= self.dict[index]['number']:
        #         continue
        #     else:
        #         return False
        # return True


    def apply_constraint(self, portfolio, option = ConstraintOption.ADJUST):

        # Check if the constraint is satisfied
        if self.check_constraint(portfolio):
            return True

        # If the constraint is not satisfied, apply the constraint option
        if option == ConstraintOption.PENALIZE:
            self.penalize_individual(portfolio)
        elif option == ConstraintOption.INVALIDATE:
            self.invalidate_individual(portfolio)
        elif option == ConstraintOption.ADJUST:
            self.adjust_individual(portfolio)

        return


    def adjust_individual(self, portfolio):
        """
        Adjusts the portfolio so that the number of assets per index matches the maximum constraint.

        This method checks the difference between #assets in the portfolio and the limitthe current number of assets for each index in the portfolio
        and the required number specified in the constraint. If any index has fewer assets than required,
        it swaps an asset from the most overrepresented index to the most underrepresented index.
        The process is repeated recursively until all indices meet their asset count constraints.

        Args:
            portfolio (Portfolio): The portfolio object to be adjusted.

        Returns:
            None: The portfolio is modified in place to satisfy the index asset constraints.
    """

        # # Get #assets in the portfolio for each index
        # count = self.get_count(portfolio)
        # # For each index, get the maximum #assets allowed
        # for index in self.dict:
        #     max_allowed = self.dict[index]['number']
        #     # While the count of this index is greater than the maximum allowed
        #     while count[index] > max_allowed:
        #         # Get an asset to swap out
        #         out_asset = self.get_out_asset(portfolio, index)
        #         # Get an asset to swap in
        #         in_asset = self.get_in_asset(portfolio, index)
        #         portfolio.swap_assets(out_asset, in_asset)
        #         count = self.get_count(portfolio)  # Update counts after swap
        # return


        #Get #assets per index
        count = self.get_count(portfolio)
        diff = [count[index] - self.dict[index]['number'] for index in self.dict]
        # Difference between #assets in the portfolio and #assets in the constraint
        diff_dict = dict(zip(list(self.dict.keys()), diff))

        # Gets the index with the minimum difference
        min_index = min(diff_dict, key=diff_dict.get)
        #If the difference is negative, swap an asset with the index with the maximum difference
        if diff_dict[min_index] < 0:
            max_index = max(diff_dict, key=diff_dict.get)
            self.swap_assets(portfolio, min_index, max_index)
            self.adjust_individual(portfolio)
        # If the difference is positive, return
        else:
            return


    def swap_assets(self, portfolio, in_index, out_index):

        #Get 
        out_asset = self.get_out_asset(portfolio, out_index)
        in_asset = self.get_in_asset(portfolio, in_index)
        portfolio.swap_assets(out_asset, in_asset)

        return 


    def get_out_asset(self, portfolio, index):

        for asset in portfolio:
            if portfolio.get_object(asset).index == index:
                return asset


    def get_in_asset(self, portfolio, index):

        asset = choice(portfolio.indexes[index].asset_list)
        if asset in portfolio:
            self.get_in_asset(portfolio, index)
        return asset

        # for asset in portfolio.all_assets:
        #     if portfolio.get_asset_index(asset).name == index:
        #         if asset in portfolio:
        #             continue
        #         return asset


    def get_count(self, portfolio):

        # Count the number of assets in the portfolio for each index
        return Counter([portfolio.get_object(i).index for i in portfolio.prtf_dict])
