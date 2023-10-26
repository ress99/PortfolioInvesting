from functools import partial
import random
import os
import plotly.graph_objects as go
import logging
import time
import datetime
import json
import pickle

from deap import creator
from deap import base

import data_op as op
import config as c
from portfolio import Portfolio


class Module:

    def register(self, alias, function, *args, **kargs):

        pfunc = partial(function, *args, **kargs)
        pfunc.__name__ = alias
        pfunc.__doc__ = function.__doc__

        setattr(self, alias, pfunc)

    def get_asset_weights(self, prtf, asset_weights, random_weights):

        #Random weights
        if random_weights:
            asset_weights = [random.random() for _ in range(10)]
            prtf.asset_weights = asset_weights
            prtf.normalize_asset_weights()
        #Same weights
        elif asset_weights is None:
            prtf.apply_same_weights()
        #Specific weights
        else:
            prtf.asset_weights = asset_weights
            prtf.normalize_asset_weights()  

        return  

    def init_Portfolio_Individual(self, assets = None, asset_weights = None, random_weights = False):

        prtf = creator.Individual(list(self.indexes.values()), 
                                cardinality_constraint=self.PRTF_SIZE,
                                start_date = self.START_DATE, 
                                end_date = self.END_DATE)
        if assets is None:
            prtf.prtf_dict = random.sample(self.all_assets, self.PRTF_SIZE)
        else:
            prtf.prtf_dict = assets

        self.get_asset_weights(prtf, asset_weights, random_weights)

        #If the assets and dates are not compatible, recall with same arguments
        if not op.check_valid_dates(prtf.prtf_df, self.START_DATE, self.END_DATE):
            prtf = self.init_Portfolio_Individual(assets, asset_weights, random_weights)

        return prtf
    
    def creator_individual(self, objectives):
        creator.create("FitnessMax", base.Fitness, weights=objectives)
        creator.create("Individual", Portfolio, fitness=creator.FitnessMax)

    def delete_creator_individual(self):

        if hasattr(creator, 'FitnessMax'):
            del creator.FitnessMax
        if hasattr(creator, 'Individual'):
            del creator.Individual
        return

    def dominates(self, A, B):
        
        if A[0] >= B[0] and A[1] >= B[1] and (A[0] > B[0] or A[1] > B[1]):
            return True
        return False

    def find_non_dominant(self):
        non_dominant = []

        for i, ind_A in enumerate(self.pop):
            is_dominated = False
            for j, ind_B in enumerate(self.pop):
                if i != j:
                    if self.dominates(ind_B.fitness.wvalues, ind_A.fitness.wvalues):
                        is_dominated = True
                        break
            if not is_dominated:
                non_dominant.append(ind_A)

        return non_dominant
    
    def get_invalid_inds(self, pop):
        return [ind for ind in pop if not ind.fitness.valid]
    
    def evaluate_population(self, pop = None):

        if pop is None:
            pop = self.pop                
        list(map(self.evaluate, pop))
        return

    def plot_paretos(self):

        fig = go.Figure()
        for i in self.pareto_fronts[:-1]:
            fig.add_trace(go.Scatter(x=i[:, 0], 
                                    y=i[:, 1], 
                                    mode = 'markers + lines',
                                    marker = dict(size=5, color='black')
                                    ))
        fig.add_trace(go.Scatter(x=self.pareto_fronts[-1][:, 0], 
                                y=self.pareto_fronts[-1][:, 1], 
                                mode = 'markers + lines',
                                marker = dict(size=8, color='red')
                                ))
        fig.update_layout(title='Portfolios',
                        xaxis_title='Return',
                        yaxis_title='Variance',
                        width = 900,
                        height = 700
                        )
        fig.show()

    def read_pickle(self, filename, as_folder = None):
        
        pickle_data = self.get_pickle_data(filename, as_folder)
        my_object = self.get_data_from_pickle(pickle_data)
        return my_object

    def get_pickle_data(self, filename, folder = None):
        
        if folder is None:
            folder = self.folder
        filename_json = filename + '.json'
        folder_names = [c.prtf_folder, folder, filename_json]
        with open(os.path.join(*folder_names), 'r') as json_file:
            json_data = json.load(json_file)
        objectives = tuple(json_data['objectives'])
        self.creator_individual(objectives)

        filename_pkl = filename + '.pkl'
        folder_names = [c.prtf_folder, folder, filename_pkl]
        with open(os.path.join(*folder_names), 'rb') as file:
            pickle_data = pickle.load(file)
        
        return pickle_data

    def get_pickle_filename(self, filename = None):
        if filename is None:
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H.%M")
        f_pkl = formatted_datetime + '.pkl'
        f_json = formatted_datetime + '.json'
        return f_pkl, f_json
    
    def save_pkl_json(self, data_to_pickle, f_pkl, f_json):

        pkl_folder_names = [c.prtf_folder, self.folder, f_pkl]
        with open(os.path.join(*pkl_folder_names), 'wb') as file:
            pickle.dump(data_to_pickle, file)

        json_folder_names = [c.prtf_folder, self.folder, f_json]
        with open(os.path.join(*json_folder_names), 'w') as json_file:
            json.dump({"objectives": self.objectives}, json_file)

        return

    def save_pickle(self, filename = None):

        f_pkl, f_json = self.get_pickle_filename(filename)
        data_to_pickle = self.get_data_to_pickle()

        self.save_pkl_json(data_to_pickle, f_pkl, f_json)
        
        return
    
    def init_data_dict(self):

        init_data = {"indexes": list(self.indexes.values()), "PRTF_SIZE": self.PRTF_SIZE, "objectives": self.objectives, 
            "START_DATE": self.START_DATE, "END_DATE": self.END_DATE, 
            "black_box_filepath": self.black_box_filepath, "bb_mode": self.bb_mode, 
            "CXPB": self.CXPB, "MUTPB": self.MUTPB,
            "pop_size": self.pop_size, "generations": self.generations}
        
        return init_data

    @property
    def final_prtf(self):
        return self._final_prtf
    
    @final_prtf.setter
    def final_prtf(self, ind):

        if isinstance(ind, Portfolio):
            self._final_prtf = ind
        else:
            print('Please choose a valid individual')

    @final_prtf.deleter
    def final_prtf(self):
        if hasattr(self, '_final_prtf'):
            delattr(self, '_final_prtf')

    @property
    def indexes(self):
        return self._indexes
    
    @indexes.setter
    def indexes(self, idxs):
        self._indexes = {i.name: i for i in idxs}

    @property
    def all_assets(self):
        assets = []
        aux_list = [list(i) for i in self.indexes.values()]
        [assets.extend(i) for i in aux_list]
        return assets