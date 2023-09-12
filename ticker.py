import pandas as pd
import datetime as dt

#####
#Class to save the information regarding company stock values
#And used to calculate various stock-related indicators
#####
class Ticker:

    #Class attributes
    data_folder = 'Data/'
    initial_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    quote_columns = initial_columns[1:-1]

    #####
    #Reads the data from the excel file and
    #Saves it in self.data
    #####
    def save_all_data(self):

        data = pd.read_excel(self.filename)                              
        data['Date'] = data['Date'].dt.date                          

        return data
    

    #####
    #Returns the data between a start and end date
    #If not specified, it returns the original data.
    #If a column is specified, only that one and Date columns are returned
    ##### 
    def get_interval_data(self, column = None, start_date = None, end_date = None):
        
        #Get original data with all or one column
        if column == None:
            data = self.data
        elif column in self.quote_columns:
            data = self.data[['Date', column]]  
        else:
            print('Column not available')

        #If any start date was specified, remove any data before that 
        if start_date is not None:
            start_date_time = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
            data = data.loc[data['Date'] >= start_date_time]

        #If any start date was specified, remove any data after that 
        if end_date is not None:
            end_date_time = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
            data = data.loc[data['Date'] <= end_date_time]
        
        return data

    #####
    #Returns the stock return between start_date and end_date
    #If these are not specified, the first and last day are used
    #The column on which to calculate the returns can also be specified.
    #It is set as 'Close' by default
    #####
    def stock_return(self, column = 'Close', start_date = None, end_date = None):

        #In case of an invalid column
        if column not in Ticker.quote_columns:
            print ('Variable column is not valid')
            return

        #Gets the data on which to calculate the return
        data = self.get_interval_data(start_date, end_date)

        #If there is no data, return
        if data.empty:
            print('There is no data between these two dates. They might have been assigned in the wrong order.')
            return

        #Get first and last value and calculate the return with it
        first_value = data.iloc[0][column]
        last_value = data.iloc[-1][column]
        stock_return = 100 * (last_value - first_value) / first_value

        return stock_return

    

    def __init__(self, ticker_name):

        #Initializes ticker parameters
        self.ticker_name = ticker_name
        self.filename = self.data_folder + 'Historical Data/' + self.ticker_name + '.xlsx'
        self.data = self.save_all_data()
        self.first_day = self.data.iloc[0]['Date']
        self.last_day = self.data.iloc[-1]['Date']
    