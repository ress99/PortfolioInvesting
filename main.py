import datetime as dt
import time
import numpy as np
from ticker import Ticker
from sp500 import SP500
from portfolio import Portfolio
import math
import plotly.express as px
import pandas as pd
import logging
import data_op as op

'''logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s: %(levelname)s: %(message)s'
                    )'''

ftre = Ticker('ftre')
sp500 = SP500()
prtf = Portfolio(sp500, cardinality_constraint=10, start_date='2022-01-01', end_date='2023-01-01', column = 'Close')
prtf.prtf_dict = ['MMM']
aux1 = aux2 = None
start = time.perf_counter()
for i in sp500.tickers_list:
    aux2 = aux1
    aux1 = i
    if i == 'MMM':
        continue
    prtf.substitute_tickers(aux2, aux1)
end = time.perf_counter()
print(end - start)

prtf.prtf_dict = ['nclh', 'ftre', 'aapl']
prtf.ticker_weights = [0.5, 0.25, 0.25]
print(prtf.tickers_list, prtf.portfolio_return(), prtf.portfolio_variance())

prtf = Portfolio(sp500, cardinality_constraint=10, start_date='2022-01-01', end_date='2023-01-01', column = 'Close')
prtf.prtf_dict = ['nclh', 'hlt', 'mar', 'c', 'l', 'a', 't', 'aapl', 'ko', 'apa']
start = time.time()
prtf.apply_same_weights()
prtf_return = prtf.portfolio_return()
prtf_var = prtf.portfolio_variance()
end = time.time()
print(end - start)
'''
prtf.prtf_dict = ['nclh', 'hlt', 'mar', 'c', 'l', 'a', 't', 'aapl', 'ko', 'apa']
prtf.apply_same_weights()
prtf.portfolio_return(start_date = '2023-01-01', end_date = '2023-08-08')
data = prtf.get_tickers_same_df(how_merge='inner', column='Close')
'''
'''
var = df[prtf.tickers_list].var()
cov = df[prtf.tickers_list].cov()
corr = df[prtf.tickers_list].corr()


print(std[0]**2)
print(std[1]**2)
cov1 = math.sqrt(var[0]) * math.sqrt(var[1]) * 0.934536
cov2 = std[0] * std[1] * 0.934536
'''

'''print(cov1, cov2)
print(cov)
print(corr.loc['CCL', 'NCLH'])'''

data = {
    'Stock1': [100, 105, 110, 108, 115],
    'Stock2': [50, 52, 55, 54, 58],
    'Stock3': [80, 85, 88, 87, 90],
}
df = pd.DataFrame(data)
weights = np.array([0.3, 0.4, 0.4])  # Corresponding to Stock1, Stock2, Stock3, ...


prtf.prtf_dict = ['rcl', 'ccl', 'nclh', 'ko', 'aapl']
prtf.ticker_weights = [0.3, 0.4, 0.4]
all_tickers_df = prtf.get_tickers_same_df(start_date = '2023-08-15', how_merge = 'inner')


def portfolio_variance():
    covariance_matrix = df.cov()
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
    portfolio_risk = np.sqrt(portfolio_variance)
    return portfolio_variance

def calculate_portfolio_variance(prtf, df):

    total = 0
    std = all_tickers_df[prtf.tickers_list].std()
    cov = all_tickers_df[prtf.tickers_list].cov()
    corr = all_tickers_df[prtf.tickers_list].corr()


    weights = prtf.ticker_weights
    print(weights)
    total = 0

    for idx1, ticker_name1 in enumerate(prtf.tickers_list):
        total += (std[idx1] ** 2) * (weights[idx1] ** 2)
        print('Added contribution for ', ticker_name1)
        for idx2, ticker_name2 in enumerate(prtf.tickers_list):
            if idx2 <= idx1:
                continue
            cov12_ = cov.loc[ticker_name1, ticker_name2]
            cov12 = std[idx1] * std[idx2] * corr.loc[ticker_name1, ticker_name2]
            print(round(cov12_, 2), round(cov12, 2), round(cov12_, 2) == round(cov12, 2))

            total += 2 * weights[idx1] * weights[idx2] * cov12
            print('Added joint contribution of ', ticker_name1, 'and ', ticker_name2)
    print(total)


prtf.apply_same_weights()
all_tickers_df = prtf.get_tickers_same_df(start_date = '2023-08-15', how_merge = 'inner')


def get_all_tickers(sp500):
    start = time.time()
    j = 0
    for i in sp500.tickers_list:
        starttt = time.time()
        sp500.get_ticker(i)
        enddd = time.time() 
        print(enddd - starttt, enddd - start, j)
        j +=1
    end = time.time()   
    print(end - start)

get_all_tickers(sp500)


import heapq

returns = []
tickers = []
for i in sp500.tickers_list:
    sp500.get_ticker(i)
    aux = sp500.sp500_dict[i].stock_return(column = 'Close', start_date = START_DATE, end_date = END_DATE)
    print(i, aux)
    returns.append(aux)
    tickers.append(i)

top_ten = heapq.nlargest(10, range(len(returns)), key=lambda i: returns[i])