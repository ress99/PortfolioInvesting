import os
import data_op as op
import pandas as pd
import numpy as np
import talib as ta
from module import Module
import logging
import time
import config as c

import random
import json


class Portfolio_Optimization(Module):

    attributes_list = ['assets', 'pop', 'pareto_front', 'pareto_fronts', 'final_prtf']


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
                 pop_size = None, generations = None, bb_path = None, filename = None):

        self.folder = c.po_folder

        if filename is not None:
            self.init_from_file(filename)
        else:
            self.init_attributes(indexes, prtf_size, objectives, start_date, end_date, 
                                 bb_path, CXPB, MUTPB, pop_size, generations, filename)
       

    def init_population(self):

        self.pop = [self.init_Portfolio_Individual
                    (assets = self.assets, asset_weights= None, random_weights = True) 
                    for _ in range(self.pop_size)]
        return

    def set_assets(self, assets = None, pkl_filename = None, json_filename = None):

        if assets:
            self.assets = assets
        elif pkl_filename:
            pickle_data = self.get_pickle_data(pkl_filename, folder = c.as_folder)
            self.assets = self.set_assets_from_pickle(pickle_data)
        elif json_filename:
            self.assets = self.set_assets_from_json(json_filename)
        
        else:
            print("Please set a filename or an asset list.")
    
    def set_assets_from_pickle(self, pickle_data):
        if 'final_prtf' in pickle_data:
            final_prtf = pickle_data['final_prtf']
            return final_prtf.asset_list
        print('There is no final Portfolio in this pickle')
        return []

    def set_assets_from_json(self, json_filename):
        folder_names = [c.prtf_folder, c.assets_folder, json_filename + '.json']
        with open(os.path.join(*folder_names), 'rb') as file:
            json_data = json.load(file)

        return json_data