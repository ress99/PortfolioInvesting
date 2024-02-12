import datetime as dt
import time
import numpy as np

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
from memory_profiler import memory_usage, profile


from deap import base
from deap import creator
from deap import tools

from asset_selection import Asset_Selection
from portfolio_optimization import Portfolio_Optimization
from constraints import ConstraintOption, Minimum_Weight, Index_Asset_Limit
from index import SP500, DAX40
from ticker import SP_Ticker, DAX_Ticker

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
    as_sel.plot_objective_space()


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
    # as_sel.evaluate = e.return_risk
    # as_sel.select = tools.selNSGA2
    # as_sel.algorithm = a.base_algorithm

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
    # po.set_assets(as_sel.final_prtf.asset_list) 
    # po.mate = m.po_mate
    # po.mutate = mut.po_reverse_weights
    # po.evaluate = e.return_risk
    # po.select = tools.selNSGA2
    # po.algorithm = a.base_algorithm

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
    # po.set_assets(as_sel.final_prtf.asset_list) 
    # po.mate = m.po_mate
    # po.mutate = mut.po_reverse_weights
    # po.evaluate = e.return_risk
    # po.select = tools.selNSGA2
    # po.algorithm = a.base_algorithm

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
    # po.set_assets(as_sel.final_prtf.asset_list) 
    # po.mate = m.po_mate
    # po.mutate = mut.po_reverse_weights
    # po.evaluate = e.return_risk
    # po.select = tools.selNSGA2
    # po.algorithm = a.base_algorithm

    # po.init_population()
    # po.algorithm(po)
    # po.save_to_pickle('200_generations_return_risk_high_risk')

    as_sel = Asset_Selection(filename = 'as_vs_po_return_risk')
    as_sel.final_prtf = as_sel.pareto_front[0]
    po = Portfolio_Optimization(filename = 'as_vs_po_return_risk_low_risk')
    po.plot_objective_space()
    po.plot_2_objectives_as_sel_vs_po(as_sel, po, 'Portfolio Lowest Risk')
    po.plot_all_returns(start_date = '2013-01-01', end_date = '2013-12-31')

    
    as_sel = Asset_Selection(filename = 'as_vs_po_return_risk')
    as_sel.final_prtf = as_sel.pareto_front[len(as_sel.pareto_front) // 2]
    po = Portfolio_Optimization(filename = 'as_vs_po_return_risk_medium_risk')
    po.plot_objective_space()
    po.plot_2_objectives_as_sel_vs_po(as_sel, po, 'Portfolio Median Risk')
    po.plot_all_returns(start_date = '2013-01-01', end_date = '2013-12-31')


    as_sel = Asset_Selection(filename = 'as_vs_po_return_risk')
    as_sel.final_prtf = as_sel.pareto_front[-1]
    po = Portfolio_Optimization(filename = 'as_vs_po_return_risk_high_risk')
    po.plot_objective_space()
    po.plot_2_objectives_as_sel_vs_po(as_sel, po, 'Portfolio Highest Risk')
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
    # as_sel.evaluate = e.return_risk
    # as_sel.select = tools.selNSGA2
    # as_sel.algorithm = a.base_algorithm

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
    # po.set_assets(as_sel.final_prtf.asset_list) 
    
    # po.mate = m.po_mate
    # po.mutate = mut.po_reverse_weights
    # po.evaluate = e.return_risk
    # po.select = tools.selNSGA2
    # po.algorithm = a.base_algorithm

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
    # po.set_assets(as_sel.final_prtf.asset_list) 

    # min_weight = Minimum_Weight(0.05)
    # po.add_constraint(min_weight)
    
    # po.mate = m.po_mate
    # po.mutate = mut.po_reverse_weights
    # po.evaluate = e.return_risk
    # po.select = tools.selNSGA2
    # po.algorithm = a.base_algorithm

    # po.init_population()
    # po.algorithm(po)
    # po.save_to_pickle('with_0.05_weight_constraint')

    as_sel = Asset_Selection(filename = 'weight_constraint')
    po_without = Portfolio_Optimization(filename = 'without_weight_constraint')
    po_with = Portfolio_Optimization(filename = 'with_0.05_weight_constraint')

    po_without.plot_objective_space()
    as_sel.plot_2_objectives_as_sel_vs_po(as_sel, po_without, 'Portfolio Optimization Without Weight Constraint')

    po_with.plot_objective_space()
    as_sel.plot_2_objectives_as_sel_vs_po(as_sel, po_with, 'Portfolio Optimization With Weight Constraint')

    as_sel.plot_2_objectives_as_sel_vs_2po(as_sel, po_without, po_with, 'Portfolio Optimization With and Without Weight Constraint')


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
    # as_sel.evaluate = e.return_risk
    # as_sel.select = tools.selNSGA2
    # as_sel.algorithm = a.base_algorithm

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
    # as_sel.algorithm = a.base_algorithm

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
    test2.update_test_list()

    test1.plot_all_test_runs(title = 'Sharpe Ratio and Var Optimization')    
    test2.plot_all_test_runs(title = 'Return and Risk Optimization')


def plot_cache():

    sizes = [50, 100, 150, 200, 250]
    secs_no_cache = [10.51, 21.03, 30.02, 41.57, 49.73]
    secs_100_cache = [2.54, 5.15, 8.25, 10.66, 14.01]
    secs_250_cache = [2.38, 4.56, 6.97, 9.38, 12.55]
    secs_500_cache = [2.37, 4.42, 6.58, 9.01, 11.19]
    secs_1000_cache = [2.35, 4.40, 6.55, 8.67, 10.94]
    secs_1000_cache = [2.34, 4.37, 6.52, 8.89, 10.89]

    plt.plot(sizes, secs_no_cache, marker = 's', label = 'Without Cache')
    plt.plot(sizes, secs_100_cache, marker = 's', label = '100 Cache, 51MB')
    plt.plot(sizes, secs_250_cache, marker = 's', label = '250 Cache, 128MB')
    plt.plot(sizes, secs_500_cache, marker = 's', label = '500 Cache, 255MB')
    plt.plot(sizes, secs_1000_cache, marker = 's', label = '1000 Cache, 511MB')
    # plt.plot(sizes, secs_500_cache, marker = 's', label = '2000 Cache, 1022MB')


    plt.title('Seconds to Complete Each Generation')
    plt.xlabel('Population Size')
    plt.ylabel('Seconds')
    plt.legend()
    plt.grid()
    plt.show()

def plot_clone():

    secs_no_deap = [0.0032, 0.0055, 0.0080, 0.01080, 0.0183]
    secs_deap = [0.3069, 0.8868, 1.7776, 2.8286, 6.4237]
    size = [10, 20, 30, 40, 50]

    # print([secs_no_cache[i] / secs_cache[i] for i in range(len(secs_cache))])

    plt.plot(size, secs_deap, marker = 's', label = 'With DEAP')
    plt.plot(size, secs_no_deap, marker = 's', label = 'Without Deap')

    plt.title('Seconds to Clone Population')
    plt.xlabel('Population Size')
    plt.ylabel('Seconds')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()



run = Test_Run(filename = 'return_risk_20')
tracking_list = ['Index', 'Maximum Risk', 'Medium Risk', 'Low Risk', 'VaR', 'Sharpe Ratio']
# tracking_list = ['Index', 'Maximum Risk', 'Medium Risk', 'Low Risk']

run.update_test_list(tracking_list = tracking_list)
run.plot_all_test_runs(tracking_list = tracking_list)
lista = []
for i in run.test_list:
    lista.append([j.iloc[-1] for j in i['df_list']])
df = pd.DataFrame(lista, columns=tracking_list)
first_column = df.iloc[:, 0]
df.iloc[:, 1:] = df.iloc[:, 1:].sub(first_column, axis=0)
for i in range(len(run.test_list)):
    run.obj.plot_objective_space_test(start_date = run.test_list[i]['start_date'], end_date = run.test_list[i]['end_date'], pareto_front = run.optimization_list[i]['prtf_list'])


sp = SP500()
init_dict = {'generations': 150, 'pop_size': c.pop_size, 'indexes': [sp],
            'prtf_size': 20, 'start_date': '2010-01-01', 'end_date': '2012-12-31',
            'CXPB': 0.5, 'MUTPB': 0.5, 'objectives': (1, -1)}
ea_dict = {'mate': m.as_mate, 
        'mutate': mut.as_mutate_all, 
        'evaluate': e.return_risk,
        'select': tools.selNSGA2, 
        'algorithm': a.base_algorithm}
as_sel = Asset_Selection(init_dict = init_dict)
as_sel.ea_ops_from_dict(ea_dict)
test = Test_Run(as_sel, '2010-01-01', 3, 1)
test.run_next_opimization()
test.run_next_opimization()
test.run_next_opimization()
test.run_next_opimization()
test.run_next_opimization()
test.run_next_opimization()
test.run_next_opimization()
test.save_pkl('return_risk_20')
exit()


# as_sel = Asset_Selection(filename = 'test1')
# as_sel.plot_all_returns(start_date = '2011-01-01', end_date = '2013-12-31')
# as_sel.plot_objective_space_test(years = 1)
# as_sel.plot_objective_space_test(start_date = '2013-01-01', end_date = '2013-12-31')

# as_sel = Asset_Selection(filename = 'test1')
# as_sel.plot_all_returns(start_date = '2014-01-01', end_date = '2014-12-31')
# as_sel.plot_objective_space_test(years = 1)
# as_sel.plot_objective_space_test(start_date = '2014-01-01', end_date = '2014-12-31')


    
sp = SP500()
init_dict = {'generations': 25, 'pop_size': c.pop_size, 'indexes': [sp],
            'prtf_size': 10, 'start_date': '2010-01-01', 'end_date': '2012-12-31',
            'CXPB': 0.5, 'MUTPB': 0.5, 'objectives': (1, -1)}
ea_dict = {'mate': m.as_mate, 
        'mutate': mut.as_mutate_all, 
        'evaluate': e.return_risk,
        'select': tools.selNSGA2, 
        'algorithm': a.base_algorithm}

print(init_dict)
as_sel = Asset_Selection(init_dict = init_dict)
as_sel.ea_ops_from_dict(ea_dict)
as_sel.init_population()
as_sel.algorithm(as_sel)


# total_size = 0
# for key in Portfolio.cache:  # Replace `lru_cache` with your actual LRU cache object
#     df_memory = Portfolio.cache[key].memory_usage(deep=True).sum()  # Memory usage of DataFrame
#     total_size += df_memory
print(op.total_size(Portfolio.cache))
# as_sel.save_to_pickle('cache_')
exit()

test = Test_Run(as_sel, '2010-01-01', 3, 1)
test.run_next_opimization()
test.run_next_opimization()
test.run_next_opimization()
test.run_next_opimization()
test.run_next_opimization()
test.run_next_opimization()
test.run_next_opimization()
test.save_pkl('return_risk')
exit()




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
# as_sel.evaluate = e.return_risk
# as_sel.select = tools.selNSGA2
# as_sel.algorithm = a.base_algorithm

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
# as_sel.algorithm = a.base_algorithm

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
# as_sel.evaluate = e.return_risk
# as_sel.select = tools.selNSGA2
# as_sel.algorithm = a.base_algorithm

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
# as_sel.evaluate = e.return_risk
# as_sel.select = tools.selNSGA2
# as_sel.algorithm = a.base_algorithm

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
# as_sel.algorithm = a.base_algorithm

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
# as_sel.algorithm = a.base_algorithm

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
# obj.plot_objective_space(obj.pareto_fronts)
# obj.plot_2_objectives_as_sel_vs_po(as_sel, po)




# as_sel = Asset_Selection(filename = 'sp_dax')
# po = Portfolio_Optimization(filename = 'sp_dax')

# po.plot_min_max()
# po.plot_min_max_product()
# as_sel.plot_objective_space(as_sel.pareto_fronts)
# po.plot_2_objectives_as_sel_vs_po(as_sel, po)








# as_sel = Asset_Selection([sp, dax], 10, (1, -1), '2021-01-01', '2022-01-01', 0.4, 0.4, 50, 50)



# limit = Index_Asset_Limit([sp, dax], (3, 3))
# as_sel.add_constraint(limit)

# as_sel.mate = m.as_mate
# as_sel.mutate = mut.as_mutate
# as_sel.evaluate = e.return_risk
# as_sel.select = tools.selNSGA2
# as_sel.algorithm = a.base_algorithm

# as_sel.init_population()
# as_sel.algorithm(as_sel)
# # as_sel.save_to_pickle('dax')

# #po = Portfolio_Optimization(filename = 'cache')

# min_weight = Minimum_Weight()
# po = Portfolio_Optimization([sp], 10, (1, -1), '2021-01-01', '2022-01-01', 0.4, 0.4, 250, 250)
# po.set_assets(pkl_filename = 'final_prtf')
# po.add_constraint(min_weight)

# po.select = tools.selNSGA2
# po.mate = m.po_mate
# po.mutate = mut.po_mutate
# po.evaluate = e.return_risk
# po.algorithm = a.base_algorithm

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


