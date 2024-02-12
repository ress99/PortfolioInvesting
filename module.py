from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import plotly.graph_objects as go
import logging
import time
import datetime
import importlib.util
import pickle
from collections import Counter


from deap import creator
from deap import base

import data_op as op
import config as c
from portfolio import Portfolio
from constraints import Constraint


class Module:

    def register(self, alias, function, *args, **kargs):

        pfunc = partial(function, *args, **kargs)
        pfunc.__name__ = alias
        pfunc.__doc__ = function.__doc__

        setattr(self, alias, pfunc)

    def get_asset_weights(self, prtf, asset_weights, random_weights):

        # #Random weights
        # if random_weights:
        #     asset_weights = [random.random() for _ in range(len(prtf.asset_weights))]
        #     prtf.asset_weights = asset_weights
        #     prtf.normalize_asset_weights()
        # #Same weights
        # elif asset_weights is None:
        #     prtf.apply_same_weights()
        # #Specific weights
        # else:
        #     prtf.asset_weights = asset_weights
        #     prtf.normalize_asset_weights()  

        #Specific weights
        if asset_weights is not None:
            prtf.asset_weights = asset_weights
            prtf.normalize_asset_weights()  

        #Random weights
        if random_weights:
            asset_weights = [random.random() for _ in range(len(prtf.asset_weights))]
            prtf.asset_weights = asset_weights
            prtf.normalize_asset_weights()

        #Same weights
        else:
            prtf.apply_same_weights()



        return  

    def init_Portfolio_Individual(self, assets = None, asset_weights = None, random_weights = False, start_date = None, end_date = None, indexes = None):

        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
        if indexes is None:
            indexes = self.index_objects

        prtf = Portfolio(indexes, cardinality_constraint=self.prtf_size, start_date = start_date, end_date = end_date)
        prtf.fitness = self.objectives

        if assets is None:
            prtf.prtf_dict = random.sample(self.all_assets, self.prtf_size)
        else:
            prtf.prtf_dict = assets

        self.get_asset_weights(prtf, asset_weights, random_weights)

        return prtf

    def dominates(self, A, B):

        is_strictly_better_in_at_least_one_dimension = False
        for a, b in zip(A, B):
            if a < b:
                return False
            elif a > b:
                is_strictly_better_in_at_least_one_dimension = True
                
        return is_strictly_better_in_at_least_one_dimension

    def find_non_dominant(self):
        non_dominant = []
        #Get population as input parameter

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
    
    def get_pop(self, pop):

        if pop is None:
            pop = self.pop   
        return pop

    def evaluate_population(self, pop = None):

        pop = self.get_pop(pop)
        list(map(self.evaluate, pop))

        return
    
    def clone_population(self, pop = None):

        pop = self.get_pop(pop)
        return [self.clone(ind) for ind in pop]

    def asset_list_counter(self, pop = None):

        pop = self.get_pop(pop)
        counter = Counter([tuple(sorted(i.asset_list)) for i in pop])
        
        return counter

    def apply_constraints(self, pop = None):

        pop = self.get_pop(pop)
        for constraint in self.constraints:
            [constraint.apply_constraint(ind) for ind in pop]
        return self.remove_invalids(pop)

    def remove_invalids(self, pop = None):

        return [ind for ind in pop if not ind.invalid]

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

    def read_pickle(self, filename, folder):
        
        if folder not in [c.as_folder, c.po_folder]:
            print('Invalid folder')
            return

        pickle_data = self.get_pickle_data(filename, folder)
        
        return self.get_data_from_pickle(pickle_data)


    def get_pickle_data(self, filename, folder = None):
        
        filename_pkl = filename + '.pkl'
        folder_names = [c.prtf_folder, folder, filename_pkl]
        with open(os.path.join(*folder_names), 'rb') as file:
            pickle_data = pickle.load(file)
        
        return pickle_data
    
    def save_pkl(self, data_to_pickle, f_pkl):

        pkl_folder_names = [c.prtf_folder, self.folder, f_pkl]
        with open(os.path.join(*pkl_folder_names), 'wb') as file:
            pickle.dump(data_to_pickle, file)

        return

    def save_to_pickle(self, filename = None):

        f_pkl = op.get_pickle_filename(filename)
        data_to_pickle = self.get_data_to_pickle()

        self.save_pkl(data_to_pickle, f_pkl)
        
        return
    
    def get_data_to_pickle(self):

        init_data = self.init_data_to_dict()
        ea_ops = self.ea_ops_to_dict()
        
        data_to_pickle = {"init_data": init_data, 'ea_ops': ea_ops}
        data_to_pickle = self.add_attributes_to_dict(self.attributes_list, data_to_pickle)
        
        return data_to_pickle
    
    def add_attributes_to_dict(self, attributes, data_to_pickle):

        for attr in attributes:
            if hasattr(self, attr):
                data_to_pickle[attr] = getattr(self, attr)
        return data_to_pickle

    def get_attributes_from_dict(self, attributes, pickle_data):
        for attr in attributes:
            if attr in pickle_data:
                setattr(self, attr, pickle_data[attr])

    def init_data_to_dict(self):

        init_data = {"indexes": list(self.indexes.values()), "prtf_size": self.prtf_size, "objectives": self.objectives, 
            "start_date": self.start_date, "end_date": self.end_date, 
            "bb_path": self.bb_path, "bb_mode": self.bb_mode, 
            "CXPB": self.CXPB, "MUTPB": self.MUTPB,
            "pop_size": self.pop_size, "generations": self.generations,
            "filename": self.filename}
        
        return init_data
    
    def init_data_from_dict(self, dictionary):

        indexes = dictionary['indexes']
        prtf_size = dictionary['prtf_size']
        objectives = dictionary['objectives']
        start_date = dictionary['start_date']
        end_date = dictionary['end_date']
        if 'pop_size' in dictionary.keys():
            pop_size = dictionary['pop_size']
        else:
            pop_size = None
        if 'generations' in dictionary.keys():
            generations = dictionary['generations']
        else:
            generations = None
        if 'filename' in dictionary.keys():
            filename = dictionary['filename']
        else:
            filename = None
        if 'bb_path' in dictionary.keys():
            bb_path = dictionary['bb_path']
        else:
            bb_path = None
        if 'CXPB' in dictionary.keys():
            CXPB = dictionary['CXPB']
        else:
            CXPB = None
        if 'MUTPB' in dictionary.keys():
            MUTPB = dictionary['MUTPB']
        else:
            MUTPB = None

        
        return indexes, prtf_size, objectives, start_date, end_date, bb_path, CXPB, MUTPB, pop_size, generations, filename
    
    def ea_ops_to_dict(self):
        ea_ops_data = {}

        if hasattr(self, 'mate'):
            ea_ops_data['mate'] = self.mate
        if hasattr(self, 'mutate'):
            ea_ops_data['mutate'] = self.mutate
        if hasattr(self, 'select'):
            ea_ops_data['select'] = self.select
        if hasattr(self, 'evaluate'):
            ea_ops_data['evaluate'] = self.evaluate
        if hasattr(self, 'algorithm'):
            ea_ops_data['algorithm'] = self.algorithm

        return ea_ops_data

    def ea_ops_from_dict(self, ea_dict):

        if 'ea_ops' in ea_dict:
            ea_dict = ea_dict['ea_ops']

        if 'mate' in ea_dict:
            self.mate = ea_dict['mate']
        if 'mutate' in ea_dict:
            self.mutate = ea_dict['mutate']
        if 'select' in ea_dict:
            self.select = ea_dict['select']
        if 'evaluate' in ea_dict:
            self.evaluate = ea_dict['evaluate']
        if 'algorithm' in ea_dict:
            self.algorithm = ea_dict['algorithm']

        return

    def init_attributes(self, indexes, prtf_size, objectives, start_date, end_date, 
                        bb_path, CXPB, MUTPB, pop_size, generations, filename):
            
        self.filename = filename
        self.indexes = indexes
        self.prtf_size = prtf_size
        self.start_date = start_date
        self.end_date = end_date
        self.objectives = objectives
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.pop_size = pop_size
        self.generations = generations
        self.constraints = []
        self.bb_path = bb_path
        if bb_path is not None:
            self.import_blackbox_module(bb_path)

    def init_from_file(self, filename):

        pickle_data = self.get_pickle_data(filename, self.folder)

        (indexes, prtf_size, objectives, start_date, end_date, 
         bb_path, CXPB, MUTPB, pop_size, generations, _) = self.init_data_from_dict(pickle_data['init_data'])

        self.init_attributes(indexes, prtf_size, objectives, start_date, end_date, 
                             bb_path, CXPB, MUTPB, pop_size, generations, filename)
        
        self.ea_ops_from_dict(pickle_data)
        self.get_attributes_from_dict(self.attributes_list, pickle_data)

        return


    def add_constraint(self, constraint):

        if isinstance(constraint, Constraint):
            self.constraints.append(constraint)
    
    def normalize(self, value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)
    
    def get_normalized_pareto_fronts(self):

        all_returns = [individual[0] for generation in self.pareto_fronts for individual in generation]
        max_return, min_return = max(all_returns), min(all_returns)
        all_risks = [individual[1] for generation in self.pareto_fronts for individual in generation]
        max_risk, min_risk = max(all_risks), min(all_risks)

        composite = []
        for generation in self.pareto_fronts:
            normalized_returns = [self.normalize(individual[0], min_return, max_return) for individual in generation]
            normalized_risks = [self.normalize(individual[1], min_risk, max_risk) for individual in generation]

            composite_metric = [ret * risk for ret, risk in zip(normalized_returns, normalized_risks)]
            composite.append(composite_metric)
        
        return composite

    def plot_min_max_product(self):

        composite = self.get_normalized_pareto_fronts()
        min_max = [[np.min(i), np.percentile(i, 25), np.mean(i), np.percentile(i, 75), np.max(i)] for i in composite]

        x_values = range(len(self.pareto_fronts))

        plt.plot(x_values, [i[0] for i in min_max], label='Min', color='blue')
        plt.plot(x_values, [i[2] for i in min_max], label='Mean', color='green')
        plt.plot(x_values, [i[4] for i in min_max], label='Max', color='red')

        plt.title('Minimum, Mean, and Maximum Return Values')
        plt.xlabel('Generations')
        plt.ylabel('Return')
        plt.grid()
        plt.legend()

        plt.show()

        return


    def standard_labels(self, title, xlabel, ylabel):

        if title is None:
            title = 'Objective Space'
        if xlabel is None:
            xlabel = 'Return'
        if ylabel is None:
            ylabel = 'Risk'

        return title, xlabel, ylabel

    def plot_min_max(self):

        min_max_return = [[i[0, 0], np.percentile(i[:, 0], 25), np.mean(i[:, 0]), np.percentile(i[:, 0], 75), i[-1, 0]] for i in self.pareto_fronts]
        min_max_risk = [[i[0, 1], np.percentile(i[:, 1], 25), np.mean(i[:, 1]), np.percentile(i[:, 1], 75), i[-1, 1]] for i in self.pareto_fronts]

        x_values = range(len(self.pareto_fronts))

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(x_values, [i[0] for i in min_max_return], label='Min', color='blue')
        plt.plot(x_values, [i[2] for i in min_max_return], label='Mean', color='green')
        plt.plot(x_values, [i[4] for i in min_max_return], label='Max', color='red')

        plt.title('Minimum, Mean, and Maximum Return Values')
        plt.xlabel('Generations')
        plt.ylabel('Return')
        plt.grid()
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x_values, [i[0] for i in min_max_risk], label='Min', color='blue')
        plt.plot(x_values, [i[2] for i in min_max_risk], label='Mean', color='green')
        plt.plot(x_values, [i[4] for i in min_max_risk], label='Max', color='red')

        plt.title('Minimum, Mean, and Maximum Risk Values')
        plt.xlabel('Generations')
        plt.ylabel('Risk')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()

        return
    
    def remove_bounds(self, remove = False, min_values = [0, 0], max_values = [1000, 1000]):

        if not remove:
            return self.pareto_fronts

        arr_list = [None] * len(self.pareto_fronts)
        for idx, arr in enumerate(self.pareto_fronts):
            arr_list[idx] = np.array([i for i in arr if all(min_v <= val <= max_v for val, (min_v, max_v) in zip(i, zip(min_values, max_values)))])
        return arr_list
        

    def plot_2_objectives_as_sel_vs_po(self, as_sel, po, title = None, xlabel = None, ylabel = None):

        title, xlabel, ylabel = self.standard_labels(title, xlabel, ylabel)

        as_sel_arr_list, po_arr_list = as_sel.pareto_fronts, po.pareto_fronts

        plt.plot(as_sel_arr_list[-1][:, 0], as_sel_arr_list[-1][:, 1], color = 'blue', marker = 'o', label = 'Asset Selection Pareto Front')
        plt.plot(po_arr_list[-1][:, 0], po_arr_list[-1][:, 1], color = '#8dff52', marker = 'o', label = 'Portfolio Optimization Pareto Front')

        if hasattr(as_sel, 'final_prtf'):
            prtf_point = as_sel.final_prtf.fitness.values
            plt.plot(prtf_point[0], prtf_point[1], color = 'red', marker = 'o', label = 'Asset Selection Final Portfolio')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend()
        plt.show()


    def plot_2_objectives_as_sel_vs_2po(self, as_sel, po_without, po_with, title = None, xlabel = None, ylabel = None):

        title, xlabel, ylabel = self.standard_labels(title, xlabel, ylabel)

        as_sel_arr_list, po_withtout_arr_list, po_with_arr_list = as_sel.pareto_fronts, po_without.pareto_fronts, po_with.pareto_fronts

        plt.plot(as_sel_arr_list[-1][:, 0], as_sel_arr_list[-1][:, 1], color = 'blue', marker = 'o', label = 'Asset Selection Pareto Front')
        plt.plot(po_withtout_arr_list[-1][:, 0], po_withtout_arr_list[-1][:, 1], color = '#8dff52', marker = 'o', label = 'Portfolio Optimization Pareto Front No Constraint')
        plt.plot(po_with_arr_list[-1][:, 0], po_with_arr_list[-1][:, 1], color = '#FAFE18', marker = 'o', label = 'Portfolio Optimization Pareto Front 0.05 Constraint')

        if hasattr(as_sel, 'final_prtf'):
            prtf_point = as_sel.final_prtf.fitness.values
            plt.plot(prtf_point[0], prtf_point[1], color = 'red', marker = 'o', label = 'Asset Selection Final Portfolio')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend()
        plt.show()


    def plot_objective_space(self, arr_list = None, title = None, xlabel = None, ylabel = None):

        title, xlabel, ylabel = self.standard_labels(title, xlabel, ylabel)

        if arr_list is None:
            arr_list = self.pareto_fronts

        plt.figure()
        label = 'All Pareto Fronts'
        for idx, arr in enumerate(arr_list):
            if idx == len(arr_list) - 1:
                plt.plot(arr[:, 0], arr[:, 1], color = 'blue', marker = 'o', label = 'Final Pareto Front')
            else:       
                plt.plot(arr[:, 0], arr[:, 1], color = '#50ef1f', marker = 'o', label = label, alpha = 0.8)
                label = None

        if hasattr(self, 'final_prtf'):
            prtf_point = self.final_prtf.fitness.values
            plt.plot(prtf_point[0], prtf_point[1], color = 'red', marker = 'o', label = 'Final Portfolio')
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend()
        plt.show()

    def plot_objective_space_test(self, years = None, months = None, start_date = None, end_date = None, pareto_front = None, title = None, xlabel = None, ylabel = None):

        if pareto_front is None:
            pareto_front = self.pareto_front

        if years is not None or months is not None:
            start_date = op.add_years_months_days(pareto_front[0].end_date, 0, 0, 1)
            end_date = op.add_years_months_days(start_date, years, months)

        elif start_date is None and end_date is None:
            return

        print(start_date, end_date)
        title, xlabel, ylabel = self.standard_labels(title, xlabel, ylabel)

        prtf_list = []
        for old_prtf in pareto_front:

            prtf = self.init_Portfolio_Individual(assets = old_prtf.asset_list, 
                                                  asset_weights = old_prtf.asset_weights,
                                                  start_date = start_date, end_date = end_date, 
                                                  indexes = old_prtf.index_objects)
            prtf_list.append(prtf)
            

        returns = [i.portfolio_return() for i in prtf_list]
        MDDs = [i.MDD() for i in prtf_list]
        plt.scatter(MDDs, returns, color = 'blue', marker = 'o', label = 'Pareto Front')
        # for idx in range(len(returns)):
        #     if idx == 1:
        #         plt.plot(MDDs[idx], returns[idx], color = 'blue', marker = 'o', label = 'Pareto Front')
        #     else:
        #         plt.plot(MDDs[idx], returns[idx], color = 'blue', marker = 'o')

        index_prtf = prtf_list[0].get_index_portfolio(prtf_list[0].index_objects, start_date, end_date)
        plt.scatter(index_prtf.MDD(), index_prtf.portfolio_return(), color = 'red', marker = 'o', label = 'Index')

        plt.xlabel('Maximum Drawdown')
        plt.ylabel('Return')
        plt.title('Pareto Front Test Run')
        plt.legend()
        plt.grid(True)
        plt.show()



    def plot_all_returns(self, pareto_front = None, start_date = None, end_date = None):

        if pareto_front is None:
            pareto_front = self.pareto_front
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        #TODO: Clone
        first_label = True
        for old_prtf in pareto_front:
            prtf = Portfolio(list(old_prtf.indexes.values()), cardinality_constraint=old_prtf.cardinality_constraint, start_date = start_date, end_date = end_date)
            prtf.prtf_dict = old_prtf.asset_list
            prtf.asset_weights = old_prtf.asset_weights
            df_to_plot = prtf.get_prtf_return_df()
            if first_label:
                plt.plot(df_to_plot.index, df_to_plot.values, label = 'Portfolios', color = '#007be0', alpha = 0.8)
                first_label = False
            else:
                plt.plot(df_to_plot.index, df_to_plot.values, color = '#007be0', alpha = 0.65)

        index_prtf = old_prtf.get_index_portfolio(list(self.indexes.values()), start_date = start_date, end_date = end_date)
        df_to_plot = index_prtf.get_prtf_return_df()
        plt.plot(df_to_plot.index, df_to_plot.values, label = 'Index', color = 'red')


        plt.xlabel('Date')
        plt.ylabel('Return')
        # plt.title('Test Run Returns')
        plt.xticks(rotation=45)
        plt.subplots_adjust(bottom=0.2)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_pareto_fronts_evolution(self, generations, arr_list = None):

        if arr_list is None:
            arr_list = self.pareto_fronts

        generations_array = [arr_list[i] for i in generations]
        colors = ['#add8e6', '#87ceeb', '#00bfff', '#1e90ff', '#0000cd']

        for idx, arr in enumerate(generations_array):
            plt.plot(arr[:, 0], arr[:, 1], color = colors[idx], marker = 'o', label = f"{generations[idx] + 1}th Generation")

        plt.title('Objective Space')
        plt.xlabel('Return')
        plt.ylabel('Risk')
        plt.grid()
        plt.legend()
        plt.show()


    def import_blackbox_module(self, module_path):
    
        if module_path == "":
            return -1
        spec = importlib.util.spec_from_file_location("blackbox_module", module_path)
        self.bb = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(self.bb)
        except FileNotFoundError as e:
            print(f"Error: {e}")

    def run_method(self, method, *args, **kwargs):

        if hasattr(self.bb, method):
            # Get the method from self.bb
            method_to_call = getattr(self.bb, method)
            # Call the method with the provided arguments and keyword arguments
            return method_to_call(*args, **kwargs)
        else:
            raise AttributeError(f"{method} not found in bb.")
        
    def set_bb_algorithm(self):

        if hasattr(self, 'bb'):
            self.algorithm = self.bb.algorithm
        else:
            print('Please select a BlackBox module.')

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
    def index_objects(self):
        return list(self._indexes.values())

    @property
    def all_assets(self):
        assets = []
        aux_list = [list(i) for i in self.indexes.values()]
        [assets.extend(i) for i in aux_list]
        return assets
    
    @property
    def bb_mode(self):
        if self.bb_path is None:
            return False
        else: 
            return True