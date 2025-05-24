"""Asset Selection class with implementation of parent's methods and attributes"""


import config as c
from module import Module
import data_op as op
from portfolio_optimization import PortfolioOptimization

# import logging
# import time
# import datetime
# import random
# import data_op as op
# import pandas as pd
# import numpy as np
# import talib as ta
# import pickle
# from functools import partial
# import os
# import json
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go


# pylint: disable=super-init-not-called
class AssetSelection(Module):
    """ Class with implementation of Module methods and attributes"""

    #Attributes for all Asset Selection objects
    attributes_list = ['pop', 'pareto_front', 'pareto_fronts', 'final_prtf']


    def __init__(self,
                 indexes = None,
                 prtf_size = None,
                 objectives = None,
                 start_date = None,
                 end_date = None,
                 CXPB = None,
                 MUTPB = None,
                 pop_size = None,
                 generations = None,
                 bb_path = None,
                 init_dict = None,
                 filename = None):
        """
        Initializes an AssetSelection object.

        This constructor initializes the object using one of three methods:
        1. If a filename is provided, the attributes are loaded from a pickle file.
        2. If init_dict is provided, the attributes are extracted from it.
        3. If no filename or dictionary is provided, the class attributes are used

        Args:
            Initialization attributes.

        Returns:
            None: The method initializes the object's attributes in place.
        """

        #Use Asset Selection folder for object
        self.folder = c.as_folder

        #If a filename is provided, extract the attributes from the pickle file
        if filename is not None:
            self.init_from_file(filename)
            return

        #If an initialization dictionary is provided, extract the attributes from it
        if init_dict is not None:
            (indexes, prtf_size, objectives, start_date, end_date,
             bb_path, CXPB, MUTPB, pop_size, generations, _
             ) = self.get_init_data_from_dict(init_dict)

        #If no filename or dictionary is provided, initialize the attributes from the arguments
        self.init_attributes(indexes, prtf_size, objectives, start_date, end_date,
                             bb_path, CXPB, MUTPB, pop_size, generations, filename)


    def get_data_from_pickle(self, pickle_data):
        """
        Creates new Asset Selection object with attributes from pickle file.

        Extracts initialization attributes, evolutionary operators,
        and additional attributes, and sets them on the new object.

        Args:
            pickle_data (dict): Dictionary containing data from the pickle file.

        Returns:
            None: The method modifies the object's attributes in place.
        """

        #Extracts initialization variables from a pickle dictionary
        (indexes, prtf_size, objectives, start_date, end_date,
         bb_path, CXPB, MUTPB, pop_size, generations, _
         ) = self.get_init_data_from_dict(pickle_data['init_data'])

        #Create new Asset Selection object with initialization attributes
        as_sel = AssetSelection(indexes, prtf_size, objectives, start_date, end_date,
                              CXPB, MUTPB, pop_size, generations, bb_path)
        #Store evolutionary operators on Asset Selection object
        as_sel.ea_ops_from_dict(pickle_data)
        #Store additional attributes on self object
        as_sel.set_attributes_from_pickle_dict(self.attributes_list, pickle_data)

        return as_sel


    def init_population(self):
        """Initialize the population of individuals for the asset selection algorithm."""
        self.pop = [self.init_portfolio_individual() for _ in range(self.pop_size)]


    def clone(self, ind):
        """Clone an individual for the asset selection algorithm."""

        #Get asset list from individual
        assets = ind.asset_list
        #Initialize portfolio object with given assets and same weights
        return self.init_portfolio_individual(assets = assets)


    def create_portfolio_optimization(self):
        """
        Creates a new PortfolioOptimization object using initialization data from the self object.

        This method extracts the initialization data from the current AssetSelection instance
        Uses it to create and return a new PO object with the same configuration.

        Returns:
            PortfolioOptimization: A new PO object initialized with the current object's data.
        """
        #Extract initialization data from self object
        init_data = self.init_data

        #Create a new PortfolioOptimization object with initialization data
        new_po = PortfolioOptimization(init_dict = init_data)

        #Return the new PortfolioOptimization object
        return new_po


    def create_test_asset_selection(self, years = None, months = None, days = None):
        """
        Creates a new AssetSelection object for testing with a shifted time period.

        This method generates a new AssetSelection instance using the initialization data
        from the current object, but with the start and end dates shifted forward by the
        specified number of years, months, and days. The new object will have a population
        initialized with the same assets and weights as the original. If the original object
        has a Pareto front or a final portfolio, these are also copied to the new object.

        Args:
            years (int, optional): Number of years as a test period
            months (int, optional): Number of months as a test period
            days (int, optional): Number of days as a test period

        Returns:
            AssetSelection: A new AssetSelection object with updated dates
            and copied population, pareto front, and final portfolio if present.
        """
        #If no time period is specified, default to 1 year
        if years is None and months is None and days is None:
            years = 1

        #Extract initialization data from the original AssetSelection object
        init_data = self.init_data

        #Update start and end dates for the new AssetSelection object
        init_data['start_date'] = init_data['end_date']
        init_data['end_date'] = op.add_years_months_days(init_data['end_date'], years, months, days)

        #Create a new AssetSelection object with the updated initialization data
        new_as_sel = AssetSelection(init_dict = init_data)

        #Initialize new population with individuals from the original population
        #Same assets and weights; new start and end dates
        new_as_sel.pop = [new_as_sel.init_portfolio_individual
                    (assets = ind.asset_list, asset_weights = ind.asset_weights)
                    for ind in self.pop]

        #If the original AssetSelection object has a pareto front, copy it to the new object
        if hasattr(self, 'pareto_front'):
            #Get the indexes of the original pareto front individuals in the original population
            old_pareto_indexes = [self.pop.index(ind) for ind in self.pareto_front]
            #Get the pareto individuals according to theirindexes in the old population
            new_as_sel.pareto_front = [new_as_sel.pop[idx] for idx in old_pareto_indexes]

        #If the original AssetSelection object has a final portfolio, copy it to the new object
        if hasattr(self, 'final_prtf'):
            #Get the index of the original final portfolio in the original population
            old_final_prtf_index = self.pop.index(self.final_prtf)
            #Get the final portfolio according to its index in the old population
            new_as_sel.final_prtf = new_as_sel.pop[old_final_prtf_index]

        #Return the new AssetSelection object
        return new_as_sel
