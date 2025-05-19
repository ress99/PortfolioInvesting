"""Portfolio Optimization class with implementation of parent's methods and attributes"""

import os
import json

import config as c
from module import Module


# import data_op as op
# import pandas as pd
# import numpy as np
# import talib as ta
# import logging
# import time
# import config as c
# import random


# pylint: disable=super-init-not-called
class PortfolioOptimization(Module):
    """Class with implementation of Module methods and attributes"""

    #Attributes for all Portfolio Optimization objects
    attributes_list = ['assets', 'pop', 'pareto_front', 'pareto_fronts', 'final_prtf']


    # def init_from_file(self, filename):

    #     pickle_data = self.get_pickle_raw_data(filename, self.folder)

    #     (indexes, prtf_size, objectives, start_date, end_date,
    #      bb_path, CXPB, MUTPB, pop_size, generations, _) = self.get_init_data_from_dict(pickle_data)

    #     self.init_attributes(indexes, prtf_size, objectives, start_date, end_date,
    #                          bb_path, CXPB, MUTPB, pop_size, generations, filename)

    #     self.ea_ops_from_dict(pickle_data)
    #     self.get_attributes_from_dict(self.attributes_list, pickle_data)

    #     return


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
                 filename = None):
        """
        Initializes a PortfoliiOptimization object.

        This constructor initializes the object using one of three methods:
        1. If a filename is provided, the attributes are loaded from a pickle file.
        2. If not, the attributes are initialized using the provided arguments.

        Args:
            Initialization attributes.

        Returns:
            None: The method initializes the object's attributes in place.
        """

        #Use Portfolio Optimization folder for object
        self.folder = c.po_folder

        #If a filename is provided, extract the attributes from the pickle file
        if filename is not None:
            self.init_from_file(filename)
            return
        #If no filename, initialize the attributes from the arguments
        self.init_attributes(indexes, prtf_size, objectives, start_date, end_date,
                                 bb_path, CXPB, MUTPB, pop_size, generations, filename)
        self.assets = None

    def init_population(self):
        """Initialize the population of individuals for the portfolio optimization algorithm."""

        self.pop = [self.init_portfolio_individual
                    (assets = self.assets,
                     asset_weights= None,
                     random_weights = True)
                    for _ in range(self.pop_size)]
        return

    def set_assets(self, assets = None, pkl_filename = None, json_filename = None):
        """
        Sets the assets for the portfolio optimization process.

        This method allows the user to set the assets in one of three ways:
        1. Directly by providing a list of assets.
        2. By loading assets from a pickle file.
        3. By loading assets from a JSON file.

        Args:
            assets (list, optional): A list of asset names to set. Defaults to None.
            pkl_filename (str, optional): The name of the pickle file (without the extension) 
                                        from which to load the assets. Defaults to None.
            json_filename (str, optional): The name of the JSON file (without the extension) 
                                        from which to load the assets. Defaults to None.

        Returns:
            None: The method modifies the `self.assets` attribute in place.
        """

        #If assets are provided as a list, set them
        if assets:
            self.assets = assets
        #If a pickle filename is provided, extract the assets from the pickle file
        elif pkl_filename:
            pickle_data = self.get_pickle_raw_data(pkl_filename, folder = c.as_folder)
            self.assets = self.get_assets_from_pickle(pickle_data)
        #If a JSON filename is provided, extract the assets from the JSON file
        elif json_filename:
            self.assets = self.set_assets_from_json(json_filename)

        else:
            print("Please set a filename or an asset list.")


    def get_assets_from_pickle(self, pickle_data):
        """
        Gets the assets from a pickle file and returns it as a list.

        Checks if the pickle data contains the final portfolio.
        If it does, extracts the asset list from the final portfolio.
        If not, returns an empty list.

        Args:
            pickle_data (dict): Dictionary containing data from the pickle file.

        Returns:
            list: List of assets extracted (if available) or an empty list.
        """

        #Check if the pickle data contains the final portfolio
        #Assets will be extracted from this portfolio
        if 'final_prtf' in pickle_data:
            final_prtf = pickle_data['final_prtf']
            return final_prtf.asset_list
        print('There is no final Portfolio in this pickle')
        return []


    def set_assets_from_json(self, json_filename):
        """
        Gets the assets from a JSON file and returns it as a list.

        Opens file and loads json data

        Args:
            json_filename (str): The name of the JSON file

        Returns:
            list: List of assets extracted from the JSON file.
        """

        #Get the path to the JSON file
        folder_names = [c.prtf_folder, c.assets_folder, json_filename + '.json']
        #Open the JSON file and load its data
        with open(os.path.join(*folder_names), 'rb') as file:
            json_data = json.load(file)

        return json_data

    def clone(self, ind):
        """Clone an individual for the asset selection algorithm."""

        #Get asset list from individual
        assets = ind.asset_list
        #Initialize portfolio object with given assets and same weights
        return self.init_portfolio_individual(assets = assets)


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

        # #Create new Asset Selection object with initialization attributes
        as_sel = PortfolioOptimization(indexes, prtf_size, objectives, start_date, end_date,
                              CXPB, MUTPB, pop_size, generations, bb_path)
        # #Store evolutionary operators on Asset Selection object
        # as_sel.ea_ops_from_dict(pickle_data)
        # #Store additional attributes on self object
        # as_sel.get_attributes_from_dict(self.attributes_list, pickle_data)

        return as_sel
