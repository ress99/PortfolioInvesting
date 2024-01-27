import pandas as pd
import datetime as dt
import config as c
import data_op as op

class Asset:

    #####
    #Reads the data from the csv file and
    #Saves it in self.data
    #####
    def save_hist_data(self, all_data = False):

        data = pd.read_csv(self.filename)   

        if all_data is False:
            first_day = c.first_day
        else:
            first_day = '1900-01-01'

        data = data.loc[data['Date'] > first_day]         
        data.reset_index(drop = True)  
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

        return data
  
    def get_price(self, date, column = None):

        #Get original data with all or one column
        if column == None:
            data = self.data
        elif column in op.quote_columns:
            data = self.data[['Date', column]]  
        else:
            print('Column not available')

        return op.get_price_on_date(data, date)


    #####
    #Returns the asset return between start_date and end_date
    #If these are not specified, the first and last day are used
    #The column on which to calculate the returns can also be specified.
    #It is set as 'Close' by default
    #####
    def asset_return(self, column = 'Close', start_date = None, end_date = None):

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
        asset_return = 100 * (last_value - first_value) / first_value

        return asset_return

    @property
    def first_day(self):
        return self.self.data.iloc[0]['Date']

    @property
    def last_day(self):
        return self.data.iloc[-1]['Date']

    @property
    def total_days(self):
        return (self.last_day - self.first_day).days


#####
#Class to save the information regarding company stock values
#And used to calculate various stock-related indicators
#####
class SP_Ticker(Asset):

    index = 'SP500'

    #####
    #Reads the data from the csv file and
    #Saves it in self.fdata
    #####
    def save_fin_data(self):

        filename = op.get_path('SP500', 'F', self.ticker_name + c.filetype)       
        data = pd.read_csv(filename, index_col = 0)

        return data

    def get_financial_value(self, financial, iloc_value = 0):

        value = self.fdata.iloc[iloc_value][financial]
        if isinstance(value, str):
            if value == '-':
                return -1
            value = int(value.replace(',', ''))
    
        return value


    def __init__(self, ticker_name, fin_data = False):

        #Initializes ticker parameters
        self.ticker_name = ticker_name
        self.filename = op.get_path(self.index, 'H', self.ticker_name + c.filetype)
        self.data = self.save_hist_data()
        if fin_data:
            self.fdata = self.save_fin_data()

    def price_book_ratio(self):

        equity = self.get_financial_value('Total Equity Gross Minority Interest')
        shares = self.get_financial_value('Ordinary Shares Number')
        bv_per_share = equity / shares
        self.pb_ratio = self.data.iloc[-1]['Close'] / bv_per_share
        
        return
    
    def return_on_equity(self):

        net_income = self.get_financial_value('Net Income Common Stockholders')
        equity = self.get_financial_value('Total Equity Gross Minority Interest')
        self.roe = 100 * (net_income / equity)

        return
    
    def price_earnings_ratio(self):

        eps = self.get_financial_value('Basic EPS')
        self.pe = self.data.iloc[-1]['Close'] / eps

        return


class DAX_Ticker(Asset):
    
    index = 'DAX40'

    def __init__(self, ticker_name):

        #Initializes ticker parameters
        
        self.ticker_name = ticker_name
        self.filename = op.get_path(self.index, 'H', self.ticker_name + c.filetype)
        self.data = self.save_hist_data()
