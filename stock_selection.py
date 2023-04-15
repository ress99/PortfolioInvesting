import data_op as op
import pandas as pd
import numpy as np
import talib as ta

def random_stock_selection (n_stocks):
    
    sp500_tickers = op.get_sp500_tickers()
    len_sp500 = len(sp500_tickers)
    if n_stocks > len_sp500:
        n_stocks = len_sp500


    #return ['ECL', 'CAH', 'TECH']
    return ['ECL', 'CAH', 'TECH', 'TXT', 'MRNA', 'TJX', 'ZBRA', 'ACN', 'CARR', 'SCHW']
    #return random.sample(sp500_tickers, n_stocks)

def get_portfolio_stocks(n_stocks):

    return random_stock_selection(n_stocks)

tsla = op.get_ticker('TSLA')

output = ta.SMA(tsla['Close'])
print(ta.get_functions())
print(output)

'''dict = {'portfolio': asd}  
       
df = pd.DataFrame(dict) 
    
# saving the dataframe 
df.to_csv('asd.csv') '''