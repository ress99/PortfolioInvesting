import time
import logging
import data_op as op
from deap import tools
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd

from asset_selection import Asset_Selection
from portfolio_optimization import Portfolio_Optimization
from constraints import ConstraintOption, Minimum_Weight, Index_Asset_Limit
from index import SP500, DAX40
from portfolio import Portfolio


import mate as m
import mutate as mut
import evaluate as e
import select_ as s
import algorithm as a
import config as c

logger = logging.getLogger(__name__)


class Test_Run():

    def run_next_opimization(self, save_file = False):

        start_date, end_date = self.get_train_start_end_dates()
        self.obj.start_date = start_date
        self.obj.end_date = end_date

        self.obj.init_population()
        self.obj.algorithm(self.obj)

        self.update_train_test_lists(self.obj)

        if save_file:
            self.obj.save_to_pickle()

        return self.obj

    def update_train_test_lists(self, obj):

        # prtf_list = self.get_prtfs_list(obj.pareto_front)
        aux_dict = {'prtf_list': obj.pareto_front, 'start_date': self.obj.start_date, 'end_date': self.obj.end_date}
        self.optimization_list.append(aux_dict)
        # self.calculate_test_list(len(self.optimization_list) - 1)

    def get_best_sharpe_ratio_prtf(self, pareto_front):

        sharpe_list = [i.sharpe_ratio() for i in pareto_front]
        return pareto_front[sharpe_list.index(max(sharpe_list))]
    
    def get_best_var_prtf(self, pareto_front):

        var_list = [i.VaR() for i in pareto_front]
        return pareto_front[var_list.index(min(var_list))]
        

    def get_prtfs_list(self, pareto_front, tracking_list):

        prtf_list = []

        for i in tracking_list:
            if i == 'Maximum Risk':
                prtf_list.append(pareto_front[-1])
            elif i == 'Medium Risk':
                prtf_list.append(pareto_front[len(pareto_front) // 2])
            elif i == 'Low Risk':
                prtf_list.append(pareto_front[0])
            elif i == 'Sharpe Ratio':
                prtf_list.append(self.get_best_sharpe_ratio_prtf(pareto_front))
            elif i == 'VaR':
                prtf_list.append(self.get_best_var_prtf(pareto_front))
            elif i == 'Index':
                prtf_list.append(pareto_front[-1].get_index_portfolio(list(pareto_front[0].indexes.values())))

        return prtf_list
    
    def get_tracking_list(self, tracking_list = None):

        if tracking_list is None:
            tracking_list = ['Index', 'Maximum Risk', 'Medium Risk', 'Low Risk']

        return tracking_list

    def calculate_test_list(self, index, tracking_list = None):

        tracking_list = self.get_tracking_list(tracking_list)

        if index > len(self.optimization_list):
            return
        
        old_end_date = self.optimization_list[index]['end_date']
        start_date = old_end_date
        end_date = op.add_years_months_days(old_end_date, years = self.test_years)

        original_prtf_list = self.get_prtfs_list(self.optimization_list[index]['prtf_list'], tracking_list)
        prtf_list = []
        df_list = []

        for idx, portfolio in enumerate(original_prtf_list):
            prtf = Portfolio(list(portfolio.indexes.values()), cardinality_constraint=portfolio.cardinality_constraint, start_date = start_date, end_date = end_date)
            prtf.prtf_dict = portfolio.asset_list
            prtf.apply_same_weights()
            one_test_df = prtf.get_prtf_return_df()
            prtf_list.append(prtf)
            df_list.append(one_test_df)
            
        # prtf = prtf.get_index_portfolio(list(portfolio.indexes.values()))
        # one_test_df = prtf.get_prtf_return_df()

        # prtf_list.append(prtf)
        # df_list.append(one_test_df)

        return {'prtf_list': prtf_list, 'df_list': df_list, 'start_date': start_date, 'end_date': end_date, 'tracking_list': tracking_list}
        
    def update_test_list(self, tracking_list = None):

        self.test_list = []
        for i in range(len(self.optimization_list)):
            aux_dict = self.calculate_test_list(i, tracking_list)
            self.test_list.append(aux_dict)
        
        return


    def run_portfolio_optimization(self, file, save_file = False, constraint = True):

        po = Portfolio_Optimization(self.indexes, self.prtf_size, self.objectives, self.start_date, self.end_date, 0.6, 0.6, self.pop_size, self.generations)
        po.get_assets(pkl_filename = file)

        if constraint:
            min_weight = Minimum_Weight(0.05)
            po.add_constraint(min_weight)

        po.select = tools.selNSGA2
        po.mate = m.po_mate
        po.mutate = mut.po_mutate
        po.evaluate = e.evaluate
        po.algorithm = a.basic

        po.init_population()
        po.algorithm(po)

        if save_file:
            po.save_to_pickle(file)

        return po

    def init_attributes(self, obj, start_date, train_years, test_years, optimization_list = None, test_list = None):

        self.obj = obj
        self.start_date = start_date
        self.train_years = train_years
        self.test_years = test_years
        self.optimization_list = optimization_list if optimization_list is not None else []
        self.test_list = test_list if test_list is not None else []

    def __init__(self, obj = None, start_date = None, train_years = None, test_years = None, filename = None):

        if filename is not None:
            self.import_pkl(filename)
            return 
        
        self.init_attributes(obj, start_date, train_years, test_years)


    def get_train_start_end_dates(self):

        if len(self.optimization_list) == 0:
            start_date = self.start_date
            end_date = op.add_years_months_days(start_date, years = self.train_years)
        else:
            start_date = op.add_years_months_days(self.optimization_list[-1]['start_date'], years = self.test_years)
            end_date = op.add_years_months_days(self.optimization_list[-1]['end_date'], years = self.test_years)

        return start_date, end_date

    def plot_all_test_runs(self, tracking_list = None):

        if len(self.test_list) == 0:
            self.update_test_list(tracking_list)

        plt.figure(figsize = (12, 6))
        for idx in range(len(self.test_list[0]['df_list'])):
            df_to_plot = self.append_dfs([j['df_list'][idx] for j in self.test_list])
            plt.plot(df_to_plot.index, df_to_plot.values, label = self.test_list[0]['tracking_list'][idx])
        
        for list in self.test_list:
            plt.axvline(x=op.date_str_to_dt(list['end_date']), color='black', linestyle='--', lw=1.5)
        
        plt.xlabel('X Axis Label')
        plt.ylabel('Y Axis Label')
        plt.title('Sine Wave Plot')
        plt.legend()
        plt.grid(True)
        plt.show()

    def append_dfs(self, df_list):

        dfs_to_concat = [df_list[0]]
        for df in df_list[1:]:
            base = dfs_to_concat[-1].iloc[-1] / 100
            based_df = df * base
            if based_df.index[0] == dfs_to_concat[-1].index[-1]:
                dfs_to_concat.append(based_df.iloc[1:])
            else:
                dfs_to_concat.append(based_df)

        result_df = pd.concat(dfs_to_concat)
        return result_df

    def plot_multiple_test_runs(self, indices, tracking_list = None):

        if len(self.test_list) == 0:
            self.update_test_list()

        length = len(indices)
        if length == 1:
            fig, axs = plt.subplots()
            axs = [axs]
        elif length == 2:
            fig, axs = plt.subplots(1, 2)
        elif length == 3:
            fig, axs = plt.subplots(1, 3, figsize = (15, 5))

        for i, index in enumerate(indices):
            self.plot_test_run(index, axs[i])

        plt.tight_layout()
        plt.show()
    
    def plot_test_run(self, index, ax):

        df_list = self.test_list[index]['df_list']

        for idx, df_to_plot in enumerate(df_list):
            ax.plot(df_to_plot.index, df_to_plot.values, label = self.test_list[index]['tracking_list'][idx])

        ax.set_xlabel('Date')
        ax.set_ylabel('Return')
        ax.set_title(f'Portfolio Returns {index + 1}')
        ax.legend()
        ax.grid(True)

        for label in ax.get_xticklabels():
            label.set_rotation(45)


    def get_test_returns(self, index, start_date = None, end_date = None):

        original_prtf_list = self.optimization_list[index]['prtf_list']

        if start_date is None and end_date is None:
            old_end_date = self.optimization_list[index]['end_date']
            start_date = op.add_years_months_days(old_end_date, days = 1)
            end_date = op.add_years_months_days(old_end_date, years = self.test_years)

        prtf_list = []
        for portfolio in original_prtf_list:
            prtf = Portfolio(list(portfolio.indexes.values()), cardinality_constraint=portfolio.cardinality_constraint, start_date = start_date, end_date = end_date)
            prtf.prtf_dict = portfolio.asset_list
            prtf.apply_same_weights() # igualar aos outros
            prtf_list.append(prtf)

        df_to_plot = prtf_list[2].get_prtf_return_df()
        plt.plot(df_to_plot.index, df_to_plot.values, label = 'High Risk')

        df_to_plot = prtf_list[1].get_prtf_return_df()
        plt.plot(df_to_plot.index, df_to_plot.values, label = 'Medium Risk')

        df_to_plot = prtf_list[0].get_prtf_return_df()
        plt.plot(df_to_plot.index, df_to_plot.values, label = 'Low Risk')
 
        prtf = original_prtf_list[0]
        prtf = Portfolio(list(portfolio.indexes.values()), cardinality_constraint=len(prtf.all_assets), start_date = start_date, end_date = end_date)
        prtf.prtf_dict = portfolio.all_assets
        prtf.apply_same_weights()

        df_to_plot = prtf.get_prtf_return_df()
        plt.plot(df_to_plot.index, df_to_plot.values, label = 'S&P500')

            
        plt.xlabel('X Axis Label')
        plt.ylabel('Y Axis Label')
        plt.title('Sine Wave Plot')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_pkl(self, filename = None):

        f_pkl = op.get_pickle_filename(filename)
        pkl_folder_names = [c.prtf_folder, c.test_folder, f_pkl]
        with open(os.path.join(*pkl_folder_names), 'wb') as file:
            pickle.dump(self, file)

        return

    def import_pkl(self, filename):
        
        f_pkl = filename + '.pkl'
        folder_names = [c.prtf_folder, c.test_folder, f_pkl]
        with open(os.path.join(*folder_names), 'rb') as file:
            test_obj = pickle.load(file)
        
        self.init_attributes(test_obj.obj, test_obj.start_date, test_obj.train_years, test_obj.test_years, test_obj.optimization_list, test_obj.test_list)
        return test_obj