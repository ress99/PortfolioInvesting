import datetime as dt
import time
import numpy as np
from index import SP500, DAX40
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

from deap import base
from deap import creator
from deap import tools

from stock_selection import Stock_Selection
from portfolio_optimization import Portfolio_Optimization

import mate as m
import mutate as mut
import evaluate as e
import select_ as s
import algorithm as a

'''log_level = logging.getLevelName(c.logging_level)
logger = logging.getLogger(__name__)
logger.setLevel(log_level)
formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')
file_handler = logging.FileHandler('download_data.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)'''

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s: %(levelname)s: %(message)s'
                    )


sp = SP500()
dax = DAX40()
sp.get_asset('TSLA', fin_data = True)
tsla = sp['TSLA']

tsla.data['Date']

print(tsla.data.Date)


ss = Stock_Selection([dax], 10, (1, -1), '2021-01-01', '2022-01-01', 0.4, 0.4, 50, 50)

ss.select = tools.selNSGA2
ss.mate = m.as_mate
ss.mutate = mut.as_mutate
ss.evaluate = e.evaluate
ss.run_algorithm = a.basic

start = time.perf_counter()

ss.init_population()
ss.run_algorithm(ss)
end = time.perf_counter()
# ss.final_prtf = ss.pareto_front[0]
ss.save_pickle()
print(end - start)


# assets = ['BG', 'FTNT', 'WRB', 'NTAP', 'KO', 'CSX', 'LVS', 'EMR', 'EL', 'WDC']
# po = Portfolio_Optimization([sp500], 10, (1, -1), '2021-01-01', '2022-01-01', 0.4, 0.4, 20, 20)
# po.get_assets(pkl_filename = '2023-10-19_16.25')

# po.select = tools.selNSGA2
# po.mate = m.po_mate
# po.mutate = mut.po_mutate
# po.evaluate = e.evaluate
# po.run_algorithm = a.basic
# start = time.perf_counter()

# po.init_population()
# po.run_algorithm(po, False)
# po.save_pickle()

# end = time.perf_counter()
# print(end - start)

