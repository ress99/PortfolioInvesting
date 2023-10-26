import pandas as pd
import numpy as np
import config as c
import datetime as dt
import os



data_folder = 'Data/'
initial_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
quote_columns = initial_columns[1:-1]


def date_str_to_dt(date):

    #print('converting string to datetime.date')
    if date is None:
        return_date = date
    else:
        try:
            dt.datetime.strptime(date, '%Y-%m-%d')
            return_date = date
        except ValueError:
            return_date = None
    
    return return_date

def df_start_to_end_date(data, start_date = None, end_date = None):

    new_data = data.copy()
    if start_date is not None:
        start_date_time = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        new_data = new_data.loc[data['Date'] >= start_date_time]

    #If any start date was specified, remove any data after that 
    if end_date is not None:
        end_date_time = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
        new_data = new_data.loc[data['Date'] <= end_date_time]

    return new_data

def check_valid_dates(data, start_date = None, end_date = None):
    
    data = df_start_to_end_date(data, start_date = start_date, end_date = end_date)
    if data.empty:
        return False
    else:
        return True

#####
#Remove all columns from the ticker's dataframe except the ones from the initial quotes dataframe
#####
def whole_df_to_quotes(df):

    column_list = []                                          #Buffer to save existing columns
    for column in quote_columns:                              #If column is in the dataframe, append to buffer
        if column in df.columns:
            column_list.append(column)
    return df[column_list]                                    #Return dataframe with the columns in the buffer



#####
#Get quote information from various tickers in the same dataframe.
#The 'column' variable decides which original column to use, and the
#'how_merge' decides how to merge, either outer or inner 
#Can be done by just creating a Portfolio
#####
def get_various_tickers (tickers, column = 'Close', how_merge = 'outer'):

    how_merge_list = ['inner', 'outer']                         #Merge options

    if column not in quote_columns:
        print ('Variable column is not valid')
        return
    
    if how_merge not in how_merge_list:
        print('Variable how_merge must be either \'outer\' or \'inner\'')
        return
        
    for tic in tickers:                                                         #For loop that passes all tickers
        to_save_filename = data_folder + 'Historical Data/' + tic + c.filetype
        data = pd.read_csv(to_save_filename)
        data['Date'] = data['Date'].dt.date

        if tic == tickers[0]:                                                   #If it's the first ticker
            hist_df = data[['Date', column]].copy()                             #Create new dataframe identical to it
            hist_df.rename(columns={column: tic}, inplace = True)               #Rename new column with name of ticker

        else:                                                                                   #For following tickers
            hist_df = hist_df.merge(data[['Date', column]], on = 'Date', how = how_merge)       #Merge dataframes with outer or inner
            hist_df.rename(columns={column: tic}, inplace = True)
    
    return hist_df
    


#####
#Concatenates tickers in the same dataframe for 
#####
def same_df_tickers (sel_tickers):

    for tic in sel_tickers:
        to_save_filename = 'Data/Historical Data/' + tic + c.filetype
        data = pd.read_csv(to_save_filename)
        if tic == sel_tickers[0]:
            hist_df = data[['Date', 'Close']].copy()
            hist_df.rename(columns={"Close": tic}, inplace = True)
        else:
            hist_df = hist_df.merge(data[['Date', 'Close']], on = 'Date')
            hist_df.rename(columns={"Close": tic}, inplace = True)
    
    return hist_df


def get_path(index_name, data_type = None, filename = None):

    folder_names = [c.data_folder, index_name]

    if data_type is not None:
        if data_type == 'H':
            folder_names.append(c.hist_folder)
        if data_type == 'F':
            folder_names.append(c.financial_folder)

    if filename is not None:
        folder_names.append(filename)

    return os.path.join(*folder_names)
