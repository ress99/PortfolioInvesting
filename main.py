import datetime as dt
import time
import numpy as np
from ticker import Ticker
from sp500 import SP500
from portfolio import Portfolio
import math
import plotly.express as px




def test_std_var(df):
    sum = 0
    asd = 'TSLA'
    mean = df[asd].mean()
    for i in range(df[asd].count()):
        sum += (mean - df[asd].iloc[i]) ** 2
    sum = sum / (df[asd].count() - 1)
    std = math.sqrt(sum)
    var = std ** 2
    print(std, var)


sp500 = SP500()

prtf = Portfolio(sp500, 10)
'''prtf.add_ticker_to_dict('ccl')
prtf.add_ticker_to_dict('nclh')
prtf.add_ticker_to_dict('hlt')
prtf.add_ticker_to_dict('mar')
prtf.add_ticker_to_dict('c')
prtf.add_ticker_to_dict('L')
prtf.add_ticker_to_dict('A')
prtf.add_ticker_to_dict('T')'''

prtf.prtf_dict = ['ccl', 'nclh']

print(prtf.tickers_list)
print(prtf.ticker_weights)

'''prtf.add_ticker_to_dict('nclh')
prtf.add_ticker_to_dict('ccl')
prtf.ticker_weights = [0.5, 0.5]

total = prtf.portfolio_return(start_date = '2010-12-12', end_date = '2019-12-12')



var = df[prtf.tickers_list].var()
cov = df[prtf.tickers_list].cov()
corr = df[prtf.tickers_list].corr()


print(std[0]**2)
print(std[1]**2)
cov1 = math.sqrt(var[0]) * math.sqrt(var[1]) * 0.934536
cov2 = std[0] * std[1] * 0.934536

print(cov1, cov2)
print(cov)
print(corr.loc['CCL', 'NCLH'])'''

prtf.ticker_weights = [0.5, 0.5]
all_tickers_df = prtf.get_tickers_same_df(how_merge = 'inner')
std = all_tickers_df[prtf.tickers_list].std() * 0.01
cov = all_tickers_df[prtf.tickers_list].cov()
corr = all_tickers_df[prtf.tickers_list].corr()

weights = prtf.ticker_weights
total = 0

def calculate_portfolio_variance():

    prtf.normalize_ticker_weights()
    all_tickers_df = prtf.get_tickers_same_df(how_merge = 'inner')
    std = all_tickers_df[prtf.tickers_list].std() * 0.01
    #std = all_tickers_df[prtf.tickers_list].std()
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
            #cov12 = cov.loc[ticker_name1, ticker_name2]
            cov12 = std[idx1] * std[idx2] * corr.loc[ticker_name1, ticker_name2]

            total += 2 * weights[idx1] * weights[idx2] * cov12
            print('Added joint contribution of ', ticker_name1, 'and ', ticker_name2)
    print(total)

#calculate_portfolio_variance()


