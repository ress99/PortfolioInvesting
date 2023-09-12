import pandas as pd
import datetime as dt
import config as c
import data_op as op

#####
#Class to save the information regarding company stock values
#And used to calculate various stock-related indicators
#####
class Ticker:

    #####
    #Reads the data from the csv file and
    #Saves it in self.data
    #####
    def save_all_data(self):

        data = pd.read_csv(self.filename)                              
        data['Date'] = pd.to_datetime(data['Date']).dt.date                    

        return data
    

    #####
    #Returns the data between a start and end date
    #If not specified, it returns the original data.
    #If a column is specified, only that one and Date columns are returned
    ##### 
    def get_interval_data(self, column = None, start_date = None, end_date = None, date = True):
        
        #Get original data with all or one column
        if column == None:
            data = self.data
        elif column in op.quote_columns:
            data = self.data[['Date', column]]  
        else:
            print('Column not available')

        #If any start date was specified, remove any data before that 
        data = op.df_start_to_end_date(data, start_date = start_date, end_date = end_date)
        
        if not date: 
            data.drop(columns = ['Date'], inplace = True)

        self.selected_data = data
        return data
    

    #####
    #Returns the stock return between start_date and end_date
    #If these are not specified, the first and last day are used
    #The column on which to calculate the returns can also be specified.
    #It is set as 'Close' by default
    #####
    def stock_return(self, column = 'Close', start_date = None, end_date = None):

        #In case of an invalid column
        if column not in op.quote_columns:
            print ('Variable column is not valid')
            return

        #Gets the data on which to calculate the return
        data = self.get_interval_data(column = None, start_date = start_date, end_date = end_date)

        #If there is no data, return
        if data.empty:
            print('There is no data between these two dates. They might have been assigned in the wrong order.')
            return 0

        #Get first and last value and calculate the return with it
        first_value = data.iloc[0][column]
        last_value = data.iloc[-1][column]
        stock_return = 100 * (last_value - first_value) / first_value

        return stock_return



    def __init__(self, ticker_name):

        #Initializes ticker parameters
        self.ticker_name = ticker_name
        self.filename = c.hist_folder + self.ticker_name + c.filetype
        self.data = self.save_all_data()
        self.selected_data = self.data.copy()
        self.first_day = self.data.iloc[0]['Date']
        self.last_day = self.data.iloc[-1]['Date']
        self.total_days = len(self.data)
    