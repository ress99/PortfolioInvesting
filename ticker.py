import pandas as pd
import time
import datetime as dt







class Ticker:

    data_folder = 'Data/'
    initial_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    quote_columns = initial_columns[1:-1]

    #####
    #
    #####
    def save_all_data(self):

        data = pd.read_excel(self.filename)                              
        data['Date'] = data['Date'].dt.date                          

        return data
    

    def get_historical_data(self, column = None):
      
        if column == None:                                                  #If nothing is specified, return whole dataframe
            return self.data[self.initial_columns]
        
        elif column in self.quote_columns:                                       #If the column value is correctly specified, 
            return self.data[['Date', column]]                                   #Return it with date column
        
        else:                                                               #If incorrectly specified, print error and return 
            print('Column name not available')
            return

    def get_interval_data(self, start_date = None, end_date = None):

        data = self.data

        if start_date is not None:
            start_date_time = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
            data = data.loc[data['Date'] >= start_date_time]

        if end_date is not None:
            end_date_time = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
            data = data.loc[data['Date'] <= end_date_time]
        
        return data

    def stock_return(self, column = 'Close', start_date = None, end_date = None):

        if column not in Ticker.quote_columns:
            print ('Variable column is not valid')
            return

        data = self.get_interval_data(start_date, end_date)

        if data.empty:
            print('There is no data between these two dates. Check if they were assigned in the right order.')
            return

        first_value = data.iloc[0][column]
        last_value = data.iloc[-1][column]
        stock_return = 100 * (last_value - first_value) / first_value

        return stock_return

    

    def __init__(self, ticker_name):

        self.ticker_name = ticker_name
        self.filename = self.data_folder + 'Historical Data/' + self.ticker_name + '.xlsx'
        self.data = self.save_all_data()
        self.first_day = self.data.iloc[0]['Date']
        self.last_day = self.data.iloc[-1]['Date']
    