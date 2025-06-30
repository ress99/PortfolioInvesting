from bs4 import BeautifulSoup
import requests
import yfinance as yf
import datetime as dt
import pandas as pd
import os
import data_op as op
import time
import config as c
import logging


from datetime import datetime, date
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


#####
#Remove all columns from the ticker's dataframe except the ones from the initial quotes dataframe
#####
def data_index_to_date(data):

    #data.index = data.index.tz_localize(None)           #Remove timezone from index
    data.reset_index(inplace = True)                    #Reset index to number of rows
    data['Date'] = data['Date'].dt.date                 #Convert 'Date' column from pandas datetime to datetime.date
    
    return data



#####
#Updates the quotes of all the assets in the index.csv file
#If a list of companies is not specified, all the assets will be updated
#####
def update_historical_data(index, assets = None):

    if assets is None:                                                     #If no list of assets is specified
        assets = index.get_all_assets()                               #Gets list of all S&P500 assets
    i = 0
    for tic in assets:
        to_save_filename = op.get_path(index.name, 'H', tic + c.filetype)
        logger.info('The next asset is: %s', tic)

        if os.path.exists(to_save_filename):                                        #If the file exists

            data = pd.read_csv(to_save_filename)                                  #Obtain dataframe from file
            date_today = dt.date.today()                                            #Gets today's date
            last_date = datetime.strptime(
                data.iloc[-1]['Date']
                , "%Y-%m-%d").date()
            last_date += dt.timedelta(days=1)                                       #Increments 1 day

            if last_date < date_today:                                                                  #If today's date is after last
                logger.info('Asset %s will be updated.', tic)                                              #date in file, then update file
                new_data = yf.download(tic, start = last_date, end = date_today, interval = '1d')       #Download new data

                if len(new_data) > 0:                                               #If downloaded data is not null
                    new_data = data_index_to_date(new_data)
                    new_df = pd.concat([data, new_data])                            #Concatenate both dataframes
                    new_df.reset_index(inplace = True, drop = True)
                    new_df.to_csv(to_save_filename, index = False)                #Save to csv
            else:
                logger.info('No update available for %s.', tic)                               #If there is no update for this asset

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



def get_index(header):

    spans = header.find_all('span')
    header_list = [span.text for span in spans[1:]]
    for idx, h in enumerate(header_list):
        if h == 'ttm':
            header_list[idx] = 'TTM'
        else:
            header_list[idx] = dt.datetime.strptime(h, '%m/%d/%Y').date()

    return header_list


def get_financials(index, assets = None):

    if assets is None:
        assets = index.get_all_assets()   

    for asset in assets:
        
        logger.info('Downloading financials from %s', asset)

        filepath = op.get_path('SP500', 'F', asset + c.filetype)
        if os.path.exists(filepath):
            continue

        headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        url_list = [f"https://finance.yahoo.com/quote/{asset}/financials",
            f"https://finance.yahoo.com/quote/{asset}/balance-sheet?p={asset}",
            f"https://finance.yahoo.com/quote/{asset}/cash-flow?p={asset}"]

        for url in url_list:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find the financial data table
                header = soup.find('div', class_='D(tbhg)')
                table = soup.find('div', class_='D(tbrg)')
                rows = table.find_all('div', class_='D(tbr)')

                list_of_lists = [None] * len(rows)
                for idx, row in enumerate(rows):

                    value = [None] * (len(row))
                    element_list = row.find_all('div', class_='D(tbc)')
                    value = [element.text for element in element_list]

                    list_of_lists[idx] = value

                list_of_lists = [list(row) for row in zip(*list_of_lists)]  

                if 'financials' in url:
                    df = pd.DataFrame(list_of_lists[1:], columns=list_of_lists[0], index = get_index(header))
                else:
                    df_to_merge = pd.DataFrame(list_of_lists[1:], columns=list_of_lists[0], index = get_index(header))
                    df = pd.merge(df, df_to_merge, right_index=True, left_index=True)

            else:
                print(f"Error: Unable to retrieve data for {asset}")
                return None
            
        df.to_csv(filepath)

        logger.info('Financials from %s saved.', asset)

    return

if __name__ == '__main__':

    print("Uncomment to update Data")
    # sp = SP500()
    # update_historical_data(sp)
    # get_financials(sp)

    # dax = DAX40()
    # update_historical_data(dax, ['^GDAXI'])
    # get_financials(dax)
