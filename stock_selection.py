import data_op as op
import pandas as pd
import numpy as np
import talib as ta
from module import Module
import importlib.util
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
from deap import base
from deap import creator
from deap import tools

class Stock_Selection(Module):

    def register(self, alias, function, *args, **kargs):

        pfunc = partial(function, *args, **kargs)
        pfunc.__name__ = alias
        pfunc.__doc__ = function.__doc__

        setattr(self, alias, pfunc)

    def get_data_to_pickle(self):

        init_data = self.init_data_dict()
        
        data_to_pickle = {"init_data": init_data}
        if hasattr(self, 'pop'):
            data_to_pickle["pop"] = self.pop
        if hasattr(self, 'pareto_front'):
            data_to_pickle["pareto_front"] = self.pareto_front
        if hasattr(self, 'pareto_fronts'):
            data_to_pickle["pareto_fronts"] = self.pareto_fronts
        if hasattr(self, 'final_prtf'):
            data_to_pickle["final_prtf"] = self.final_prtf
        
        return data_to_pickle
    
    def get_data_from_pickle(self, pickle_data):

        indexes = pickle_data['init_data']['indexes']
        PRTF_SIZE = pickle_data['init_data']['PRTF_SIZE']
        objectives = pickle_data['init_data']['objectives']
        START_DATE = pickle_data['init_data']['START_DATE']
        END_DATE = pickle_data['init_data']['END_DATE']
        black_box_filepath = pickle_data['init_data']['black_box_filepath']
        bb_mode = pickle_data['init_data']['bb_mode']
        CXPB = pickle_data['init_data']['CXPB']
        MUTPB = pickle_data['init_data']['MUTPB']
        pop_size = pickle_data['init_data']['pop_size']
        generations = pickle_data['init_data']['generations']

        ss = Stock_Selection(indexes, PRTF_SIZE, objectives, START_DATE, END_DATE,
                              CXPB, MUTPB, pop_size, generations, black_box_filepath)
        
        ss.bb_mode = bb_mode

        if 'pop' in pickle_data:
            ss.pop = pickle_data['pop']
        if 'pareto_front' in pickle_data:
            ss.pareto_front = pickle_data['pareto_front']
        if 'pareto_fronts' in pickle_data:
            ss.pareto_fronts = pickle_data['pareto_fronts']
        if 'final_prtf' in pickle_data:
            ss.final_prtf = pickle_data['final_prtf']

        return ss


    def init_population(self):
        self.creator_individual(self.objectives)
        self.pop = [self.init_Portfolio_Individual() for _ in range(self.pop_size)]
        return

    def import_blackbox_module(self, module_path):
    
        if module_path == "":
            return -1
        spec = importlib.util.spec_from_file_location("blackbox_module", module_path)
        self.bb = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(self.bb)
        except FileNotFoundError as e:
            print(f"Error: {e}")

    def run_method(self, method):
        
        if hasattr(self.bb, method):
            self.bb.run_blackbox()
            return 1
        else:
            print("Module does not have a" + method + "function.")
            return False


    def __init__(self, indexes = None, PRTF_SIZE = None, objectives = None, 
                 START_DATE = None, END_DATE = None, CXPB = None, MUTPB = None, 
                 pop_size = None, generations = None, black_box_filepath = None):

        self.folder = c.as_folder
        if indexes is None:
            return
        self.indexes = indexes
        self.PRTF_SIZE = PRTF_SIZE
        self.objectives = objectives
        self.START_DATE = START_DATE
        self.END_DATE = END_DATE
        self.black_box_filepath = black_box_filepath
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.pop_size = pop_size
        self.generations = generations
        if black_box_filepath is not None:
            self.import_blackbox_module(black_box_filepath)
            self.bb_mode = True
        else:
            self.bb_mode = False

    