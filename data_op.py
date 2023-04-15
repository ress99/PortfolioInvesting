import pandas as pd
import numpy as np


data_folder = 'Data/'
initial_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
quote_columns = initial_columns[1:-1]


#####
#Returns all S&P500 Tickers being used
#####
def get_sp500_tickers():

    tickers_df = pd.read_csv(data_folder + 'sp500tickers.csv')
    tickers = tickers_df['S&P500'].tolist()

    return tickers



#####
#Returns a dataframe with the requested data.
#If column is None, return full dataframe;
#If column is a quote column, return it with the Data;
#Else, print error and return
#####
def get_ticker(tic, column = None):

    filename = data_folder + 'Historical Data/' + tic + '.xlsx'         
    data = pd.read_excel(filename)                                      #Get the ticker's filename and read excel
    data['Date'] = data['Date'].dt.date                                 #Date column from 

    if column == None:                                                  #If nothing is specified, return whole dataframe
        return data
    
    elif column in quote_columns:                                       #If the column value is correctly specified, 
        return data[['Date', column]]                                   #Return it with date column
    
    else:                                                               #If incorrectly specified, print error and return 
        print('Column name not available')
        return



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
        to_save_filename = data_folder + 'Historical Data/' + tic + '.xlsx'
        data = pd.read_excel(to_save_filename)
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
        to_save_filename = 'Data/Historical Data/' + tic + '.xlsx'
        data = pd.read_excel(to_save_filename)
        if tic == sel_tickers[0]:
            hist_df = data[['Date', 'Close']].copy()
            hist_df.rename(columns={"Close": tic}, inplace = True)
        else:
            hist_df = hist_df.merge(data[['Date', 'Close']], on = 'Date')
            hist_df.rename(columns={"Close": tic}, inplace = True)
    
    return hist_df



def get_fa(tic):

    filename = data_folder + 'FA/' + tic + '.xlsx'
    data = pd.read_excel(filename)
    data['Date'] = data['Date'].dt.date
    return data

