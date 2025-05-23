"""Module with the for Data Operation Methods"""


import os
import sys
import random
import datetime as dt
from datetime import timedelta

import pandas as pd
# import numpy as np
from dateutil.relativedelta import relativedelta

import config as c



# data_folder = 'Data/'
initial_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
quote_columns = initial_columns[1:-1]


def get_pickle_filename(filename = None):
    """Get and return name for Pickle File"""

    if filename is None:
        now = dt.datetime.now()
        filename = now.strftime("%Y-%m-%d_%H.%M")
    f_pkl = filename + '.pkl'
    return f_pkl


def date_str_to_dt(date):
    """Gets date as String, return as Datetime.Date"""

    #print('converting string to datetime.date')
    if date is None:
        return_date = date
    else:
        try:
            return_date = dt.datetime.strptime(date, '%Y-%m-%d').date()
        except ValueError:
            return_date = None

    return return_date


def add_years_months_days(date, years = None, months = None, days = None, string = True):
    """Add Years, Months or Days to a given Date"""

    if isinstance(date, str):
        date = date_str_to_dt(date)
        if date is None:
            print('Please provide a valid Date')
            return None

    if years is None:
        years = 0
    if months is None:
        months = 0
    if days is None:
        days = 0

    new_date = date + relativedelta(years=years, months=months)
    new_date += timedelta(days=days)

    if string:
        new_date = new_date.strftime("%Y-%m-%d")

    return new_date


def df_start_to_end_date(data, start_date = None, end_date = None):
    """Cuts a Data DataFrame from Start to End Date"""

    new_data = data.copy()

    #If any start date was specified, remove any data after that
    if start_date is not None:
        start_date_time = date_str_to_dt(start_date)
        new_data = new_data.loc[data['Date'] >= start_date_time]

    #If any end date was specified, remove any data after that
    if end_date is not None:
        end_date_time = date_str_to_dt(end_date)
        new_data = new_data.loc[data['Date'] <= end_date_time]

    return new_data


def get_price_on_date(data, date, column = None):
    """
    Returns a Series with the prices on a certain Date.
    If a column is provided, returns only the corresponding value
    """

    new_data = data.copy()
    date_dt = date_str_to_dt(date)
    new_data = new_data.loc[data['Date'] >= date_dt].iloc[0]
    # new_data = new_data.loc[data['Date'] == date_dt]
    if column is not None:
        new_data = new_data[column]

    return new_data


def check_valid_dates(data, start_date = None, end_date = None):
    """Checks if the Dates provided are valid (DataFrame not empty)"""

    data = df_start_to_end_date(data, start_date = start_date, end_date = end_date)
    if data.empty:
        return False
    else:
        return True


def merge_asset_dataframes(assets, column = 'Close', how_merge = 'outer'):
    """
    Merges Asset data into the same DataFrame.
    column variable indicates the column from each asset to use.
    how_merge variable indicates the merge process.
    """

    #Merge options
    how_merge_list = ['inner', 'outer']

    #Column check
    if column not in quote_columns:
        print ('Variable column is not valid')
        return

    #Check merge option
    if how_merge not in how_merge_list:
        print('Variable how_merge must be either \'outer\' or \'inner\'')
        return

    #For each asset, retrieve data from the database
    for asset in assets:
        to_save_filename = c.data_folder + 'Historical Data/' + asset + c.filetype
        data = pd.read_csv(to_save_filename)
        data['Date'] = data['Date'].dt.date

        #If it's the first asset, create the new DataFrame and rename the columns
        if asset == assets[0]:
            hist_df = data[['Date', column]].copy()
            hist_df.rename(columns={column: asset}, inplace = True)

        #For all other assets, merge the new columns into the DataFrame
        else:
            hist_df = hist_df.merge(data[['Date', column]], on = 'Date', how = how_merge)
            hist_df.rename(columns={column: asset}, inplace = True)

    return hist_df


def get_path(index_name, data_type = None, filename = None):
    """Get the database path for a Asset"""

    #Data Folder and Index Name
    folder_names = [c.data_folder, index_name]

    #Add Historical or Financial Folder
    if data_type is not None:
        if data_type == 'H':
            folder_names.append(c.hist_folder)
        if data_type == 'F':
            folder_names.append(c.financial_folder)

    #Add File Name
    if filename is not None:
        folder_names.append(filename)

    #Join all elements of the list in a path
    return os.path.join(*folder_names)


def total_size(obj, seen=None):
    """Recursively finds size of objects in bytes."""

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum((total_size(k, seen) + total_size(v, seen)) for k, v in obj.items())
    elif hasattr(obj, '__dict__'):
        size += total_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(total_size(i, seen) for i in obj)

    return size


def aux_get_swap_numbers(lenght):
    """Auxilliary method that returns two indexes to swap in a list"""

    #Retrieves two random indexes from 0 to lenght
    idx1 = random.randint(0, lenght)
    idx2 = random.randint(0, lenght)

    #If the first index is not smaller than the first
    if idx1 >= idx2:
        #Recall the function until the condition is met
        idx1, idx2 = aux_get_swap_numbers(lenght)

    #Return the two indexes
    return idx1, idx2
