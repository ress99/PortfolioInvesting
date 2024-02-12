import data_op as op
import pandas as pd
import numpy as np
import talib as ta
from module import Module
import logging
import time
import datetime
import config as c
import pickle
from functools import partial
import os
import json

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import random

class Asset_Selection(Module):

    attributes_list = ['pop', 'pareto_front', 'pareto_fronts', 'final_prtf']

    # def get_data_to_pickle(self):

    #     init_data = self.init_data_to_dict()
    #     ea_ops = self.ea_ops_to_dict()
        
    #     data_to_pickle = {"init_data": init_data, 'ea_ops': ea_ops}
    #     data_to_pickle = self.add_attributes_to_dict(self.attributes_list, data_to_pickle)
        
    #     return data_to_pickle
    
    def get_data_from_pickle(self, pickle_data):

        (indexes, prtf_size, objectives, start_date, end_date, 
         bb_path, CXPB, MUTPB, pop_size, generations, filename) = self.init_data_from_dict(pickle_data['init_data'])

        as_sel = Asset_Selection(indexes, prtf_size, objectives, start_date, end_date,
                              CXPB, MUTPB, pop_size, generations, bb_path)
        
        as_sel.ea_ops_from_dict(pickle_data)
        as_sel.get_attributes_from_dict(self.attributes_list, pickle_data)

        return as_sel

    def init_population(self):
        self.pop = [self.init_Portfolio_Individual() for _ in range(self.pop_size)]
        return

    def clone(self, ind):

        assets = ind.asset_list
        return self.init_Portfolio_Individual(assets = assets)
    

    
    # def init_from_file(self, filename):

    #     pickle_data = self.get_pickle_data(filename, self.folder)

    #     (indexes, prtf_size, objectives, start_date, end_date, 
    #      bb_path, CXPB, MUTPB, pop_size, generations, _) = self.init_data_from_dict(pickle_data)     

    #     self.init_attributes(indexes, prtf_size, objectives, start_date, end_date, 
    #                          bb_path, CXPB, MUTPB, pop_size, generations, filename)
        
    #     self.ea_ops_from_dict(pickle_data)
    #     self.get_attributes_from_dict(self.attributes_list, pickle_data)

    #     return
    

    def __init__(self, indexes = None, prtf_size = None, objectives = None, 
                 start_date = None, end_date = None, CXPB = None, MUTPB = None, 
                 pop_size = None, generations = None, bb_path = None, init_dict = None, filename = None):

        self.folder = c.as_folder


        if init_dict is not None:
            (indexes, prtf_size, objectives, start_date, end_date, 
            bb_path, CXPB, MUTPB, pop_size, generations, _) = self.init_data_from_dict(init_dict)
        elif filename is not None:
            self.init_from_file(filename)
            return
        self.init_attributes(indexes, prtf_size, objectives, start_date, end_date, 
                             bb_path, CXPB, MUTPB, pop_size, generations, filename)
        
