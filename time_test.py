from ticker import Ticker
import time
from sp500 import SP500
from portfolio import Portfolio
import talib as ta
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import yahooquery as yq 
import datetime

def get_all_tickers():
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



sp500 = SP500()
prtf = Portfolio(sp500, 10)
prtf.add_ticker_to_dict('NVDA')

data = prtf.prtf_dict['NVDA']['ticker'].data
data.dropna(inplace = True)


sma30 = ta.SMA(data['Close'], timeperiod = 30)
ema30 = ta.EMA(data['Close'], timeperiod = 30)


fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name = 'Quote'))

fig.add_trace(go.Scatter(x=data['Date'], y=sma30, name = 'SMA_30'))
fig.add_trace(go.Scatter(x=data['Date'], y=ema30, name = 'EMA_30'))
fig.show()

mmm = yq.Ticker('MMM')
df = mmm.all_financial_data()
columns = df.columns.tolist()
cash_flow = mmm.cash_flow()
balance_sheet = mmm.balance_sheet()
balance_sheet.columns.tolist()
cash_flow.columns.tolist()

balance_sheet_columns = ['asOfDate', 'periodType','CashAndCashEquivalents', 'CommonStock', 
                         'CurrentAssets', 'CurrentDebt', 'CurrentLiabilities', 
                         'NetDebt', 'NetPPE', 'NetTangibleAssets']


print(balance_sheet[balance_sheet_columns])
print(cash_flow)
financial_data_columns = ['']
print(mmm.earning_history)
print(mmm.earnings)
print(mmm.dividend_history(start = '1900-01-01'))

