import random
from functools import partial
from random import choice
import numpy as np
import pandas as pd
import time
from enum import Enum
from collections import Counter

class ConstraintOption(Enum):
    PENALIZE = 1
    INVALIDATE = 2
    ADJUST = 3


class Constraint():

    def check_constraint(self):
        print('Abstract method')

    def apply_constraint(self):
        print('Abstract method')


class Minimum_Weight(Constraint):

    def __init__(self, threshold = 0.05):

        self.threshold = threshold

    def check_constraint(self, portfolio):
        
        weights = portfolio.asset_weights
        if min(weights) < self.threshold:
            return False
        return True

    def apply_constraint(self, portfolio, option = ConstraintOption.ADJUST):

        if self.check_constraint(portfolio):
            return True
        
        if option == ConstraintOption.PENALIZE:
            self.penalize_individual(portfolio)
        elif option == ConstraintOption.INVALIDATE:
            self.invalidate_individual(portfolio)
        elif option == ConstraintOption.ADJUST:
            self.adjust_individual(portfolio)
        
        return

    def penalize_individual(self, portfolio, penalty = 0.1):

        portfolio.penalty += penalty
        return
        

    def invalidate_individual(self, portfolio):

        portfolio.invalid = True
        return

    def adjust_individual(self, portfolio):

        del portfolio.fitness.values
        adjusted_weights = portfolio.asset_weights.copy()
        deficit = sum([self.threshold - w for w in adjusted_weights if self.threshold > w])

        surplus_assets = sorted([(i, w) for i, w in enumerate(adjusted_weights)], key=lambda x: x[1], reverse=True)

        differences = [surplus_assets[i][1] - surplus_assets[i + 1][1] for i in range(len(surplus_assets) - 1)]
        to_take_list = []

        adjusted_weights = [max(i, self.threshold) for i in adjusted_weights]
        for i in range(len(differences)):
            if ((i + 1) * differences[i]) > deficit:
                to_take = deficit / (i + 1)
            else:
                to_take = differences[i]
            deficit -= to_take * (i + 1)
            to_take_list.append(0)
            to_take_list = [i + to_take for i in to_take_list]
            if deficit < 0.000005:
                break
        for idx, value in enumerate(to_take_list):
            adjusted_weights[surplus_assets[idx][0]] -= value
                        
        portfolio.asset_weights = adjusted_weights
        
        return 
    

class Index_Asset_Limit(Constraint):

    def __init__(self, index_list, nmbr_assets_list):

        if len(index_list) != len(nmbr_assets_list):
            return False
        
        self.index_list = index_list
        self.nmbr_assets_list = nmbr_assets_list
        inside_dict = [{'object': obj, 'number': num} for i, (obj, num) in enumerate(zip(index_list, nmbr_assets_list))]
        self.dict = dict(zip([i.name for i in index_list], inside_dict))

        return

    def check_constraint(self, portfolio):
        
        if sum(self.nmbr_assets_list) > len(portfolio):
            return False

        count = self.get_count(portfolio)
        for index in self.dict:
            if count[index] >= self.dict[index]['number']:
                continue
            else:
                return False
        return True


    def apply_constraint(self, portfolio, option = ConstraintOption.ADJUST):

        if self.check_constraint(portfolio):
            return True
        
        if option == ConstraintOption.PENALIZE:
            self.penalize_individual(portfolio)
        elif option == ConstraintOption.INVALIDATE:
            self.invalidate_individual(portfolio)
        elif option == ConstraintOption.ADJUST:
            self.adjust_individual(portfolio)
        
        return

    def penalize_individual(self, portfolio, penalty = 0.1):

        portfolio.penalty += penalty
        return
        
    def invalidate_individual(self, portfolio):

        portfolio.invalid = True
        return

    def adjust_individual(self, portfolio):

        if sum(self.nmbr_assets_list) > len(portfolio.asset_list):
            return False
        
        count = self.get_count(portfolio)
        diff = [count[index] - self.dict[index]['number'] for index in self.dict]
        diff_dict = dict(zip(list(self.dict.keys()), diff))

        min_index = min(diff_dict, key=diff_dict.get)
        if diff_dict[min_index] < 0:
            max_index = max(diff_dict, key=diff_dict.get)
            self.swap_assets(portfolio, min_index, max_index)
            self.adjust_individual(portfolio)
        return 
    

    def swap_assets(self, portfolio, in_index, out_index):

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

        return Counter([portfolio.get_object(i).index for i in portfolio.prtf_dict])