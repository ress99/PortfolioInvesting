import bs4 as bs
import requests
import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import os
import data_op as op
import calendar
import ticker
import time
import config as c
import logging


from datetime import datetime, date
from urllib.request import Request, urlopen

'''log_level = logging.getLevelName(c.logging_level)
logger = logging.getLogger(__name__)
logger.setLevel(log_level)
formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')
file_handler = logging.FileHandler('download_data.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)'''


#####
#Update list of all S&P500 tickers
#####
def update_sp500_tickers():

    agent = {"User-Agent":"Mozilla/5.0"}
    url = 'https://www.slickcharts.com/sp500'
    source=requests.get(url, headers=agent).text
    df_list = pd.read_html(source)

    sp500_df = df_list[0]
    sp500_df = sp500_df.sort_values('Company')
    sp500_list = sp500_df['Symbol'].tolist()
    sp500_list = [tic.replace('.', '-') for tic in sp500_list]

    sp500_series = pd.Series (sp500_list, name = 'S&P500')
    sp500_series.to_csv('Data/sp500tickers.csv', index = False)             #Save list of all tickers to csv file

    return



#####
#Remove all columns from the ticker's dataframe except the ones from the initial quotes dataframe
#####
def data_index_to_date(data):

        #data.index = data.index.tz_localize(None)           #Remove timezone from index
        data.reset_index(inplace = True)                    #Reset index to number of rows
        data['Date'] = data['Date'].dt.date                 #Convert 'Date' column from pandas datetime to datetime.date
        
        return data



#####
#Updates the quotes of all the tickers in the sp500tickers.csv file
#If a list of companies is not specified, all the tickers will be updated
#####
def update_historical_data(tickers = None):

    if tickers is None:                                                     #If no list of tickers is specified
        tickers = op.get_sp500_tickers()                               #Gets list of all S&P500 tickers
    i = 0
    for tic in tickers:
        to_save_filename = c.hist_folder + tic + c.filetype        #Path for ticker file
        logger.info('The next ticker is: %s', tic)

        if os.path.exists(to_save_filename):                                        #If the file exists

            data = pd.read_csv(to_save_filename)                                  #Obtain dataframe from file
            date_today = dt.date.today()                                            #Gets today's date
            last_date = datetime.strptime(
                data.iloc[-1]['Date']
                , "%Y-%m-%d").date()
            last_date += dt.timedelta(days=1)                                       #Increments 1 day

            if last_date < date_today:                                                                  #If today's date is after last
                logger.info('Ticker %s will be updated.', tic)                                              #date in file, then update file
                new_data = yf.download(tic, start = last_date, end = date_today, interval = '1d')       #Download new data

                if len(new_data) > 0:                                               #If downloaded data is not null
                    new_data = data_index_to_date(new_data)
                    new_df = pd.concat([data, new_data])                            #Concatenate both dataframes
                    new_df.reset_index(inplace = True, drop = True)
                    new_df.to_csv(to_save_filename, index = False)                #Save to csv
            else:
                logger.info('No update available for %s.', tic)                               #If there is no update for this ticker

        else:   
            start = time.time()                                                            #In case the file does not exists
            data = yf.download(tic, period = 'max', interval = '1d')        #Download data from yfinance
            data = data_index_to_date(data)                                 
            data = data.iloc[:-1]                                           #Remove last row - may not be a complete day     
            data.to_csv(to_save_filename, index = False)                  #Save to csv
            end = time.time()
            logger.debug(end - start)
        logger.debug(i)
        i +=1



def download_earnings(tic):

    data_earnings = yf.Ticker(tic).earnings_history
    if data_earnings is not None:
        data_earnings['month'] = data_earnings['Earnings Date'].astype(str).str[:3].apply(lambda x: list(calendar.month_abbr).index(x))
        data_earnings['day'] = data_earnings['Earnings Date'].astype(str).str[4:6]
        data_earnings['year'] = data_earnings['Earnings Date'].astype(str).str[8:12]
        data_earnings['Date'] = pd.to_datetime(data_earnings[['year', 'month', 'day']])
        data_earnings['Date'] = data_earnings['Date'].dt.date
        data_earnings = data_earnings[['Date', 'EPS Estimate', 'Reported EPS']]
        date_today = date.today()
        data_earnings = data_earnings.loc[data_earnings['Date'] < date_today] 
    
    return data_earnings

def download_dividends(tic):

    data_dividends = yf.Ticker(tic).dividends
    data_dividends = data_dividends.to_frame()
    data_dividends = data_index_to_date(data_dividends)

    return data_dividends

def update_dividends(tic):

    to_save_filename = c.hist_folder + tic + c.filetype

    if not os.path.exists(to_save_filename):
        update_historical_data(tic)
    
    data_dividends = download_dividends(tic)


    

asd = download_dividends('ko')
asddd = yf.Ticker('ko').dividends
print(asd)


def update_financials():

    tickers = op.get_sp500_tickers()

    for tic in tickers:

        to_save_filename = c.hist_folder + tic + c.filetype
        financials_df = yf.Ticker(tic).financials.transpose()
        financials_df.reset_index(drop = False, inplace = True)
        financials_df.rename(columns={"index": "Date"})
        financials_df['Date'] = financials_df['Date'].dt.date
        if os.path.exists(to_save_filename):
            data = pd.read_csv(to_save_filename)
            data.merge(financials_df, )


'''
tic = 'AAL'
to_save_filename = c.hist_folder + tic + c.filetype
financials_df = yf.Ticker(tic).financials.transpose()
financials_df.index = financials_df.index.tz_localize(None)
financials_df.reset_index(inplace = True)
financials_df.rename(columns={"index": "Date"}, inplace = True)
financials_df['Date'] = financials_df['Date'].dt.date
fcolumns = financials_df.columns[1:]



if os.path.exists(to_save_filename):
    data = pd.read_csv(to_save_filename)
    data['Date'] = data['Date'].dt.date
    if financials_df.columns[1] not in data.columns:
        data = data.merge(financials_df, on = 'Date', how = 'outer')
        data.sort_values(by = 'Date', inplace = True)
        data.reset_index(inplace = True, drop = True)
        data.to_csv(to_save_filename, index = False)
    else:
        dates = financials_df['Date']
        for i in dates:
            #data.loc[data['Date'] == dates[i]][fcolumns] = financials_df[fcolumns]
            print(data[data['Date'] == i])
            #print(data[data['Date'] == dates[0]])
        #print(data[financials_df.columns])

'''