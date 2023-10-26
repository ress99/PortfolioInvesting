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

    def __init__(self, indexes = None, PRTF_SIZE = None, objectives = None, 
                 START_DATE = None, END_DATE = None, CXPB = None, MUTPB = None, 
                 pop_size = None, generations = None, black_box_filepath = None):

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
        self.folder = c.po_folder

    def init_population(self):
        self.creator_individual(self.objectives)
        self.pop = [self.init_Portfolio_Individual
                    (assets = self.assets, asset_weights= None, random_weights = True) 
                    for _ in range(self.pop_size)]
        return

    def get_data_to_pickle(self):
        
        init_data = self.init_data_dict()
        
        data_to_pickle = {"init_data": init_data}
        if hasattr(self, 'assets'):
            data_to_pickle["assets"] = self.assets
        if hasattr(self, 'pop'):
            data_to_pickle["pop"] = self.pop
        if hasattr(self, 'pareto_front'):
            data_to_pickle["pareto_front"] = self.pareto_front
        if hasattr(self, 'pareto_fronts'):
            data_to_pickle["pareto_fronts"] = self.pareto_fronts
        if hasattr(self, 'final_prtf'):
            data_to_pickle["final_prtf"] = self.final_prtf
        
        return data_to_pickle

    # def save_pickle(self, filename = None):

    #     f_pkl, f_json = self.get_pickle_filename(filename)
    #     data_to_pickle = self.get_data_to_pickle()
    #     self.save_pkl_json(self, data_to_pickle, f_pkl, f_json)
        
    #     return


    def get_assets_from_pickle(self, pickle_data):
        if 'final_prtf' in pickle_data:
            final_prtf = pickle_data['final_prtf']
            return final_prtf.asset_list
        print('There is no final Portfolio in this pickle')
        return []

    def get_assets_from_json(self, json_filename):

        with open(json_filename, 'rb') as file:
            json_data = json.load(file)


        return json_data['assets']

    def get_assets(self, pkl_filename = None, json_filename = None, assets = None):

        if assets:
            self.assets = assets
        elif pkl_filename:
            pickle_data = self.get_pickle_data(pkl_filename, folder = c.as_folder)
            self.assets = self.get_assets_from_pickle(pickle_data)
            self.delete_creator_individual()
        elif json_filename:
            self.assets = self.get_assets_from_json(json_filename)
        
        else:
            print("Please set a filename or an asset list.")
    


# def run(po):

#     po.pareto_fronts = [None] * po.generations

#     for g in range(po.generations):

#         # Select the offspring for the next generation
#         offspring = po.select(po.pop, len(po.pop))

#         # Clone the selected individuals
#         pop_weights = [i.asset_weights for i in offspring]
#         offspring = list(po.initPortfolio(tickers = po.assets, weights = i) for i in pop_weights)

#         for child1, child2 in zip(offspring[::2], offspring[1::2]):

#             # cross two individuals with probability CXPB
#             if random.random() < po.CXPB:
#                 po.mate(child1, child2)

#                 # fitness values of the children
#                 # must be recalculated later
#                 del child1.fitness.values
#                 del child2.fitness.values

#         for mutant in offspring:

#             # mutate an individual with probability MUTPB
#             if random.random() < po.MUTPB:
#                 po.mutate(mutant)
#                 del mutant.fitness.values

#         # Evaluate the individuals with an invalid fitness
#         invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#         po.evaluate_population(invalid_ind)

#         # The population is entirely replaced by the offspring
#         po.pop = po.select(po.pop + offspring, po.pop_size)
        
#         pareto_front = po.find_non_dominant()

#         # Gather all the fitnesses in one list and print the stats
#         pvalues = np.array([i.fitness.values for i in pareto_front])
#         pareto_front = [pareto_front[i] for i in np.argsort(pvalues[:, 0])]
#         po.pareto_fronts[g] = np.array([i.fitness.values for i in pareto_front])
#         print("Generation %d || Length of Pareto: %d ||  Mean Return is %f || Mean Variance is %f" % (g + 1, len(pareto_front), pvalues[:, 0].mean(), pvalues[:, 1].mean()))

#     po.pareto_front = pareto_front