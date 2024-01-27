import datetime as dt
import time
import numpy as np
from index import SP500, DAX40
import config as c
from portfolio import Portfolio
import math
import plotly.express as px
import pandas as pd
import logging
import data_op as op
import random
import time
import matplotlib.pyplot as plt
from functools import partial
import datetime
from statistics import median, mean
from memory_profiler import memory_usage


from deap import base
from deap import creator
from deap import tools

from asset_selection import Asset_Selection
from portfolio_optimization import Portfolio_Optimization
from constraints import ConstraintOption, Minimum_Weight, Index_Asset_Limit

from test_run import Test_Run

import mate as m
import mutate as mut
import evaluate as e
import select_ as s
import algorithm as a


random.seed(1)

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
filename = f"Logs/log_{current_time}.log"

log_level = logging.getLevelName(c.logging_level)
root_logger = logging.getLogger()
root_logger.setLevel(log_level)
formatter = logging.Formatter('%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d] %(message)s')
file_handler = logging.FileHandler(filename)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)




def pareto_front_500_generations():

    as_sel = Asset_Selection(filename = '500_generations')
    as_sel.plot_all_returns()
    as_sel.plot_all_returns(start_date = '2013-01-01', end_date = '2013-12-31')
    as_sel.plot_pareto_fronts_evolution([99, 199, 299, 399, 499])
    as_sel.plot_2_objectives()


def as_vs_po_return_risk():

    # generations = 200
    # pop_size = 100
    # indexes = [sp]
    # prtf_size = 20
    # start_date = '2010-01-01'
    # end_date = '2012-12-31'
    # objectives = (1, -1)

    # as_sel = Asset_Selection(indexes, prtf_size, objectives, start_date, end_date, 0.5, 0.5, pop_size, generations)
    # as_sel.mate = m.as_mate
    # as_sel.mutate = mut.as_mutate
    # as_sel.evaluate = e.evaluate
    # as_sel.select = tools.selNSGA2
    # as_sel.algorithm = a.basic

    # as_sel.init_population()
    # as_sel.algorithm(as_sel)
    # as_sel.save_to_pickle('200_generations_return_risk')
    # exit()
        

    # as_sel = Asset_Selection(filename = '200_generations_return_risk')
    # as_sel.final_prtf = as_sel.pareto_front[0]

    # generations = 200
    # pop_size = 100
    # indexes = [sp]
    # prtf_size = 20
    # start_date = '2010-01-01'
    # end_date = '2012-12-31'
    # objectives = (1, -1)

    # po = Portfolio_Optimization(indexes, prtf_size, objectives, start_date, end_date, 0.5, 0.5, pop_size, generations)
    # po.get_assets(as_sel.final_prtf.asset_list) 
    # po.mate = m.po_mate
    # po.mutate = mut.po_reverse_weights
    # po.evaluate = e.evaluate
    # po.select = tools.selNSGA2
    # po.algorithm = a.basic

    # po.init_population()
    # po.algorithm(po)
    # po.save_to_pickle('200_generations_return_risk_low_risk')


    # as_sel = Asset_Selection(filename = '200_generations_return_risk')
    # as_sel.final_prtf = as_sel.pareto_front[len(as_sel.pareto_front) // 2]

    # generations = 200
    # pop_size = 100
    # indexes = [sp]
    # prtf_size = 20
    # start_date = '2010-01-01'
    # end_date = '2012-12-31'
    # objectives = (1, -1)

    # po = Portfolio_Optimization(indexes, prtf_size, objectives, start_date, end_date, 0.5, 0.5, pop_size, generations)
    # po.get_assets(as_sel.final_prtf.asset_list) 
    # po.mate = m.po_mate
    # po.mutate = mut.po_reverse_weights
    # po.evaluate = e.evaluate
    # po.select = tools.selNSGA2
    # po.algorithm = a.basic

    # po.init_population()
    # po.algorithm(po)
    # po.save_to_pickle('200_generations_return_risk_medium_risk')


    # as_sel = Asset_Selection(filename = '200_generations_return_risk')
    # as_sel.final_prtf = as_sel.pareto_front[-1]

    # generations = 200
    # pop_size = 100
    # indexes = [sp]
    # prtf_size = 20
    # start_date = '2010-01-01'
    # end_date = '2012-12-31'
    # objectives = (1, -1)

    # po = Portfolio_Optimization(indexes, prtf_size, objectives, start_date, end_date, 0.5, 0.5, pop_size, generations)
    # po.get_assets(as_sel.final_prtf.asset_list) 
    # po.mate = m.po_mate
    # po.mutate = mut.po_reverse_weights
    # po.evaluate = e.evaluate
    # po.select = tools.selNSGA2
    # po.algorithm = a.basic

    # po.init_population()
    # po.algorithm(po)
    # po.save_to_pickle('200_generations_return_risk_high_risk')

    as_sel = Asset_Selection(filename = 'as_vs_po_return_risk')
    as_sel.final_prtf = as_sel.pareto_front[0]
    po = Portfolio_Optimization(filename = 'as_vs_po_return_risk_low_risk')
    po.plot_2_objectives()
    po.plot_2_objectives_as_sel_vs_po(as_sel, po)
    po.plot_all_returns(start_date = '2013-01-01', end_date = '2013-12-31')

    
    as_sel = Asset_Selection(filename = 'as_vs_po_return_risk')
    as_sel.final_prtf = as_sel.pareto_front[len(as_sel.pareto_front) // 2]
    po = Portfolio_Optimization(filename = 'as_vs_po_return_risk_medium_risk')
    po.plot_2_objectives()
    po.plot_2_objectives_as_sel_vs_po(as_sel, po)
    po.plot_all_returns(start_date = '2013-01-01', end_date = '2013-12-31')


    as_sel = Asset_Selection(filename = 'as_vs_po_return_risk')
    as_sel.final_prtf = as_sel.pareto_front[-1]
    po = Portfolio_Optimization(filename = 'as_vs_po_return_risk_high_risk')
    po.plot_2_objectives()
    po.plot_2_objectives_as_sel_vs_po(as_sel, po)
    po.plot_all_returns(start_date = '2013-01-01', end_date = '2013-12-31')


def weight_constraint():


    # generations = 200
    # pop_size = 100
    # indexes = [sp]
    # prtf_size = 10
    # start_date = '2010-01-01'
    # end_date = '2012-12-31'
    # objectives = (1, -1)

    # as_sel = Asset_Selection(indexes, prtf_size, objectives, start_date, end_date, 0.5, 0.5, pop_size, generations)
    # as_sel.mate = m.as_mate
    # as_sel.mutate = mut.as_mutate
    # as_sel.evaluate = e.evaluate
    # as_sel.select = tools.selNSGA2
    # as_sel.algorithm = a.basic

    # as_sel.init_population()
    # as_sel.algorithm(as_sel)
    # as_sel.save_to_pickle('weight_constraint')


    # generations = 200
    # pop_size = 100
    # indexes = [sp]
    # prtf_size = 10
    # start_date = '2010-01-01'
    # end_date = '2012-12-31'
    # objectives = (1, -1)

    # po = Portfolio_Optimization(indexes, prtf_size, objectives, start_date, end_date, 0.5, 0.5, pop_size, generations)
    # po.get_assets(as_sel.final_prtf.asset_list) 
    
    # po.mate = m.po_mate
    # po.mutate = mut.po_reverse_weights
    # po.evaluate = e.evaluate
    # po.select = tools.selNSGA2
    # po.algorithm = a.basic

    # po.init_population()
    # po.algorithm(po)
    # po.save_to_pickle('without_weight_constraint')


    # generations = 200
    # pop_size = 100
    # indexes = [sp]
    # prtf_size = 10
    # start_date = '2010-01-01'
    # end_date = '2012-12-31'
    # objectives = (1, -1)

    # po = Portfolio_Optimization(indexes, prtf_size, objectives, start_date, end_date, 0.5, 0.5, pop_size, generations)
    # po.get_assets(as_sel.final_prtf.asset_list) 

    # min_weight = Minimum_Weight(0.05)
    # po.add_constraint(min_weight)
    
    # po.mate = m.po_mate
    # po.mutate = mut.po_reverse_weights
    # po.evaluate = e.evaluate
    # po.select = tools.selNSGA2
    # po.algorithm = a.basic

    # po.init_population()
    # po.algorithm(po)
    # po.save_to_pickle('with_0.05_weight_constraint')

    as_sel = Asset_Selection(filename = 'weight_constraint')
    po_without = Portfolio_Optimization(filename = 'without_weight_constraint')
    po_with = Portfolio_Optimization(filename = 'with_0.05_weight_constraint')

    po_without.plot_2_objectives()
    as_sel.plot_2_objectives_as_sel_vs_po(as_sel, po_without)

    po_with.plot_2_objectives()
    as_sel.plot_2_objectives_as_sel_vs_po(as_sel, po_with)

    as_sel.plot_2_objectives_as_sel_vs_2po(as_sel, po_without, po_with)


def return_vs_sharpe_10():

    # generations = 200
    # pop_size = 100
    # indexes = [sp]
    # prtf_size = 10
    # start_date = '2010-01-01'
    # end_date = '2012-12-31'
    # objectives = (1, -1)

    # as_sel = Asset_Selection(indexes, prtf_size, objectives, start_date, end_date, 0.5, 0.5, pop_size, generations)
    # as_sel.mate = m.as_mate
    # as_sel.mutate = mut.as_mutate_all
    # as_sel.evaluate = e.evaluate
    # as_sel.select = tools.selNSGA2
    # as_sel.algorithm = a.basic

    # test = Test_Run(as_sel, '2010-01-01', 3, 1)
    # test.run_next_opimization()
    # test.run_next_opimization()
    # test.run_next_opimization()
    # test.run_next_opimization()
    # test.run_next_opimization()
    # test.save_pkl('return_risk_10')



    # generations = 200
    # pop_size = 100
    # indexes = [sp]
    # prtf_size = 10
    # start_date = '2010-01-01'
    # end_date = '2012-12-31'
    # objectives = (1, -1)

    # as_sel = Asset_Selection(indexes, prtf_size, objectives, start_date, end_date, 0.5, 0.5, pop_size, generations)
    # as_sel.mate = m.as_mate
    # as_sel.mutate = mut.as_mutate_all
    # as_sel.evaluate = e.sharpe_var
    # as_sel.select = tools.selNSGA2
    # as_sel.algorithm = a.basic

    # test = Test_Run(as_sel, '2010-01-01', 3, 1)
    # test.run_next_opimization()
    # test.run_next_opimization()
    # test.run_next_opimization()
    # test.run_next_opimization()
    # test.run_next_opimization()
    # test.save_pkl('sharpe_var_10')

    test1 = Test_Run(filename = 'sharpe_var_10')
    test2 = Test_Run(filename = 'return_risk_10')


    test1.update_test_list()
    test1.plot_all_test_runs()

    test2.update_test_list()
    test2.plot_all_test_runs()



sp = SP500()
dax = DAX40()



# pareto_front_500_generations()
as_vs_po_return_risk()
# weight_constraint()
#return_vs_sharpe_10()


# generations = 200
# pop_size = 100
# indexes = [sp]
# prtf_size = 10
# start_date = '2010-01-01'
# end_date = '2012-12-31'
# objectives = (1, -1)

# as_sel = Asset_Selection(indexes, prtf_size, objectives, start_date, end_date, 0.5, 0.5, pop_size, generations)
# as_sel.mate = m.as_mate
# as_sel.mutate = mut.as_mutate_all
# as_sel.evaluate = e.evaluate
# as_sel.select = tools.selNSGA2
# as_sel.algorithm = a.basic

# test = Test_Run(as_sel, '2010-01-01', 3, 1)
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.save_pkl('return_risk_size10')




# generations = 200
# pop_size = 100
# indexes = [sp]
# prtf_size = 20
# start_date = '2010-01-01'
# end_date = '2012-12-31'
# objectives = (1, -1)

# as_sel = Asset_Selection(indexes, prtf_size, objectives, start_date, end_date, 0.5, 0.5, pop_size, generations)
# as_sel.mate = m.as_mate
# as_sel.mutate = mut.as_mutate_all
# as_sel.evaluate = e.evaluate
# as_sel.select = tools.selNSGA2
# as_sel.algorithm = a.basic

# test = Test_Run(as_sel, '2010-01-01', 3, 1)
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.save_pkl('return_risk_size20')
# exit()




# generations = 200
# pop_size = 100
# indexes = [sp, dax]
# prtf_size = 10
# start_date = '2010-01-01'
# end_date = '2012-12-31'
# objectives = (1, -1)

# as_sel = Asset_Selection(indexes, prtf_size, objectives, start_date, end_date, 0.5, 0.5, pop_size, generations)
# as_sel.mate = m.as_mate
# as_sel.mutate = mut.as_mutate_all
# as_sel.evaluate = e.evaluate
# as_sel.select = tools.selNSGA2
# as_sel.algorithm = a.basic

# limit = Index_Asset_Limit(indexes, [5, 5])
# as_sel.add_constraint(limit)

# as_sel.init_population()
# as_sel.algorithm(as_sel)
# as_sel.save_to_pickle('sp_dax_no_constraint')




# generations = 200
# pop_size = 100
# indexes = [sp]
# prtf_size = 10
# start_date = '2010-01-01'
# end_date = '2012-12-31'
# objectives = (1, -1)

# as_sel = Asset_Selection(indexes, prtf_size, objectives, start_date, end_date, 0.5, 0.5, pop_size, generations)
# as_sel.mate = m.as_mate
# as_sel.mutate = mut.as_mutate_all
# as_sel.evaluate = e.sharpe_var
# as_sel.select = tools.selNSGA2
# as_sel.algorithm = a.basic

# test = Test_Run(as_sel, '2010-01-01', 3, 1)
# test.run_next_opimization()
# test.save_pkl('test_sharpe')


# generations = 200
# pop_size = 100
# indexes = [sp]
# prtf_size = 10
# start_date = '2010-01-01'
# end_date = '2012-12-31'
# objectives = (1, -1)

# as_sel = Asset_Selection(indexes, prtf_size, objectives, start_date, end_date, 0.5, 0.5, pop_size, generations)
# as_sel.mate = m.as_mate
# as_sel.mutate = mut.as_mutate_all
# as_sel.evaluate = e.evaluate
# as_sel.select = tools.selNSGA2
# as_sel.algorithm = a.basic

# test = Test_Run(as_sel, '2010-01-01', 3, 1)
# test.run_next_opimization()
# test.save_pkl('test_return')



tracking_list = ['Index', 'Maximum Risk', 'Medium Risk', 'Low Risk', 'VaR', 'Sharpe Ratio']

test_r = Test_Run(filename = 'test_return')
test_s = Test_Run(filename = 'test_sharpe')

test_r.update_test_list(tracking_list = tracking_list)
test_s.update_test_list(tracking_list = tracking_list)

test_r.plot_all_test_runs(tracking_list = tracking_list)
test_s.plot_all_test_runs(tracking_list = tracking_list)


test2 = Test_Run(filename = 'sharpe_var_20')
test1 = Test_Run(filename = 'sharpe_var_10')

test1.update_test_list(tracking_list = tracking_list)
test2.update_test_list(tracking_list = tracking_list)

test1.plot_all_test_runs(tracking_list = tracking_list)
test2.plot_all_test_runs(tracking_list = tracking_list)



# test = Test_Run(as_sel, '2010-01-01', 3, 1)
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.save_pkl('return_risk_20')





# generations = 200
# pop_size = 100
# indexes = [sp]
# prtf_size = 20
# start_date = '2010-01-01'
# end_date = '2012-12-31'
# objectives = (1, -1)

# as_sel = Asset_Selection(indexes, prtf_size, objectives, start_date, end_date, 0.5, 0.5, pop_size, generations)
# as_sel.mate = m.as_mate
# as_sel.mutate = mut.as_mutate
# as_sel.evaluate = e.evaluate
# as_sel.select = tools.selNSGA2
# as_sel.algorithm = a.basic

# test = Test_Run(as_sel, '2010-01-01', 3, 1)
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# # test.run_next_opimization()
# # test.run_next_opimization()
# test.save_pkl('return_risk_10')
# exit()

# generations = 200
# pop_size = 100
# indexes = [sp]
# prtf_size = 10
# start_date = '2010-01-01'
# end_date = '2012-12-31'
# objectives = (1, -1)

# as_sel = Asset_Selection(indexes, prtf_size, objectives, start_date, end_date, 0.5, 0.5, pop_size, generations)
# as_sel.mate = m.as_mate
# as_sel.mutate = mut.as_mutate
# as_sel.evaluate = e.sharpe_var
# as_sel.select = tools.selNSGA2
# as_sel.algorithm = a.basic

# test = Test_Run(as_sel, '2010-01-01', 3, 1)
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.save_pkl('sharpe_var_10')


# generations = 200
# pop_size = 100
# indexes = [sp]
# prtf_size = 20
# start_date = '2010-01-01'
# end_date = '2012-12-31'
# objectives = (1, -1)

# as_sel = Asset_Selection(indexes, prtf_size, objectives, start_date, end_date, 0.5, 0.5, pop_size, generations)
# as_sel.mate = m.as_mate
# as_sel.mutate = mut.as_mutate
# as_sel.evaluate = e.sharpe_var
# as_sel.select = tools.selNSGA2
# as_sel.algorithm = a.basic

# test = Test_Run(as_sel, '2010-01-01', 3, 1)
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.run_next_opimization()
# test.save_pkl('sharpe_var_20')



















# def wrapper_function():
#     run_asset_selection(True, False)

# if __name__ == '__main__':
#     max_memory = max(memory_usage(wrapper_function))

#     print(f"Maximum memory usage: {max_memory} MiB")


# obj = Asset_Selection(filename = 'sp_dax')
    
# test = Test(filename = 'asd')
# test = test.import_pkl('2024-01-20_17.32')
# test.plot_test_run(0)
# test.plot_test_run(1)
# test.plot_test_run(2)
# test.plot_test_run(3)


test = Test_Run(filename = 'run')
# test.get_best_sharpe_ratio_prtf(test.obj.pareto_front)
# test.get_best_var_prtf(test.obj.pareto_front)

test.plot_multiple_test_runs([-1])
exit()

# start_date, end_date = test.get_train_start_end_dates()
# test.run_asset_selection(start_date, end_date, False, False)     
# start_date, end_date = test.get_train_start_end_dates()
# test.run_asset_selection(start_date, end_date, False, False)
# start_date, end_date = test.get_train_start_end_dates()
# test.run_asset_selection(start_date, end_date, False, False)
# start_date, end_date = test.get_train_start_end_dates()
# test.run_asset_selection(start_date, end_date, False, False)
# start_date, end_date = test.get_train_start_end_dates()
# test.run_asset_selection(start_date, end_date, False, False)
# start_date, end_date = test.get_train_start_end_dates()
# test.run_asset_selection(start_date, end_date, False, False)

# test.save_pkl('test')

test.plot_all_test_runs()

test.plot_test_run(0)
test.plot_test_run(1)
test.plot_test_run(2)


# obj = run_asset_selection(True, False)
# size = op.total_size(list(obj.indexes.values()))
# print(size)

# get_test_returns(obj, years = 1)





# obj.plot_min_max()
# obj.plot_min_max_product()
# obj.plot_2_objectives(obj.pareto_fronts)
# obj.plot_2_objectives_as_sel_vs_po(as_sel, po)




# as_sel = Asset_Selection(filename = 'sp_dax')
# po = Portfolio_Optimization(filename = 'sp_dax')

# po.plot_min_max()
# po.plot_min_max_product()
# as_sel.plot_2_objectives(as_sel.pareto_fronts)
# po.plot_2_objectives_as_sel_vs_po(as_sel, po)








# as_sel = Asset_Selection([sp, dax], 10, (1, -1), '2021-01-01', '2022-01-01', 0.4, 0.4, 50, 50)



# limit = Index_Asset_Limit([sp, dax], (3, 3))
# as_sel.add_constraint(limit)

# as_sel.mate = m.as_mate
# as_sel.mutate = mut.as_mutate
# as_sel.evaluate = e.evaluate
# as_sel.select = tools.selNSGA2
# as_sel.algorithm = a.basic

# as_sel.init_population()
# as_sel.algorithm(as_sel)
# # as_sel.save_to_pickle('dax')

# #po = Portfolio_Optimization(filename = 'cache')

# min_weight = Minimum_Weight()
# po = Portfolio_Optimization([sp], 10, (1, -1), '2021-01-01', '2022-01-01', 0.4, 0.4, 250, 250)
# po.get_assets(pkl_filename = 'final_prtf')
# po.add_constraint(min_weight)

# po.select = tools.selNSGA2
# po.mate = m.po_mate
# po.mutate = mut.po_mutate
# po.evaluate = e.evaluate
# po.algorithm = a.basic

# po.init_population()

# po.algorithm(po)
# po.save_to_pickle('cache')





# import os
# import glob
# total_lines = 0
# for filepath in glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), '*.py')):
#     with open(filepath, 'r', encoding='utf-8') as file:
#         total_lines += sum(1 for line in file)
# print(total_lines)


