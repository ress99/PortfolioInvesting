"""Asset Selection class with implementation of parent's methods and attributes"""


import config as c
from module import Module

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
        1. If init_dict is provided, the attributes are extracted from it.
        2. If a filename is provided, the attributes are loaded from a pickle file.
        3. If neither is provided, the attributes are initialized using the provided arguments.

        Args:
            Initialization attributes.

        Returns:
            None: The method initializes the object's attributes in place.
        """

        #Use Asset Selection folder for object
        self.folder = c.as_folder

        #If an initialization dictionary is provided, extract the attributes from it
        if init_dict is not None:
            (indexes, prtf_size, objectives, start_date, end_date,
             bb_path, CXPB, MUTPB, pop_size, generations, _
             ) = self.get_init_data_from_dict(init_dict)
        #If a filename is provided, extract the attributes from the pickle file
        elif filename is not None:
            self.init_from_file(filename)
            return
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
        as_sel.get_attributes_from_dict(self.attributes_list, pickle_data)

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


    # def init_from_file(self, filename):

    #     pickle_data = self.get_pickle_raw_data(filename, self.folder)

    #     (indexes, prtf_size, objectives, start_date, end_date,
    #      bb_path, CXPB, MUTPB, pop_size, generations, _) = self.get_init_data_from_dict(pickle_data)

    #     self.init_attributes(indexes, prtf_size, objectives, start_date, end_date,
    #                          bb_path, CXPB, MUTPB, pop_size, generations, filename)

    #     self.ea_ops_from_dict(pickle_data)
    #     self.get_attributes_from_dict(self.attributes_list, pickle_data)

    #     return


    # def get_data_to_pickle(self):

    #     init_data = self.init_data_to_dict()
    #     ea_ops = self.ea_ops_to_dict()

    #     data_to_pickle = {"init_data": init_data, 'ea_ops': ea_ops}
    #     data_to_pickle = self.add_obj_attributes_to_dict(self.attributes_list, data_to_pickle)

    #     return data_to_pickle
