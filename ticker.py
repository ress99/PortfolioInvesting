"""Module with the Asset Class and its Subclasses"""

# import datetime as dt
import pandas as pd
import config as c
import data_op as op

class Asset:
    """Class representing an Asset"""


    @property
    def filename(self):
        """File name property, must be implemented by child classes."""
        raise NotImplementedError("Subclasses must define 'filename'.")


    @property
    def data(self):
        """Data property, must be implemented by child classes."""
        raise NotImplementedError("Subclasses must define 'data'.")


    @property
    def first_day(self):
        """First Day property."""
        return self.data.iloc[0]['Date']

    @property
    def last_day(self):
        """Last Day property."""
        return self.data.iloc[-1]['Date']

    @property
    def total_days(self):
        """Total Days property, returns the total number of days in the data."""
        return (self.last_day - self.first_day).days

    def get_historical_data(self, all_data = False):
        """Function reading historical data from the database"""

        data = pd.read_csv(self.filename)

        if all_data is False:
            first_day = c.first_day
            last_day = c.last_day
        else:
            first_day = '1900-01-01'
            last_day = '2100-01-01'

        data = data.loc[data['Date'] > first_day]
        data = data.loc[data['Date'] < last_day]
        data.reset_index(drop = True)
        data['Date'] = pd.to_datetime(data['Date']).dt.date

        return data


    def get_interval_data(self, column = None, start_date = None, end_date = None, date = True):
        """
        Returns data between an interval of two dates.
        If no start or end date is specified, the method returns all the available data.
        If a column is specified, the method returns only the Date with that column.
        """

        #Get original data with all or one column
        if column is None:
            data = self.data
        elif column in op.quote_columns:
            data = self.data[['Date', column]]  
        else:
            print('Column not available')
            return None

        #If any start date was specified, remove any data before that 
        data = op.df_start_to_end_date(data, start_date = start_date, end_date = end_date)

        if not date:
            data.drop(columns = ['Date'], inplace = True)

        return data

    def get_date_price(self, date, column = None):
        """
        Returns the price of the asset on a specified date.
        If no start or end date is specified, the method returns all the available data.
        If a column is specified, the method returns only the Date with that column.
        """

        #Get original data with all or one column
        if column is None:
            data = self.data
        elif column in op.quote_columns:
            data = self.data[['Date', column]]  
        else:
            print('Column not available')
            return None

        return op.get_price_on_date(data, date)


    def asset_return(self, column = 'Close', start_date = None, end_date = None):
        """
        Calculates the asset return between a start and end date.
        If these are not specified, the first and last days are used.
        The column on which to calculate the returns can also be specified ('Close' by default).
        """

        #In case of an invalid column
        if column not in op.quote_columns:
            print ('Variable column is not valid')
            return None

        #Gets the data on which to calculate the return
        data = self.get_interval_data(column = None, start_date = start_date, end_date = end_date)

        #If there is no data, return
        if data.empty:
            print('No data between these two dates.')
            return 0

        #Get first and last value and calculate the return
        first_value = data.iloc[0][column]
        last_value = data.iloc[-1][column]
        asset_return = 100 * (last_value - first_value) / first_value

        return asset_return


class SPTicker(Asset):
    """Class implementing the SP500 Assets"""


    index = 'SP500'


    @property
    def filename(self):
        """Overrides filename property in Asset Class"""
        return self._filename


    @property
    def data(self):
        """Overrides data property in Asset Class"""
        return self._data


    @property
    def asset_name(self):
        """Asset Name property"""
        return self._asset_name


    @property
    def fin_data(self):
        """Financial Data property"""
        return self._fin_data


    @property
    def pb_ratio(self):
        """Price to Book ratio property."""
        if not hasattr(self, '_pb_ratio'):
            raise ValueError("Financial data not available. Please initialize with 'fin_data'.")
        return self._pb_ratio


    @property
    def roe(self):
        """Return on Equity property."""
        if not hasattr(self, '_roe'):
            raise ValueError("Financial data not available. Please initialize with 'fin_data'.")
        return self._roe


    @property
    def pe(self):
        """Price to Earnings ratio property"""
        if not hasattr(self, '_pe'):
            raise ValueError("Financial data not available. Please initialize with 'fin_data'.")
        return self._pe


    def __init__(self, asset_name, fin_data = False):

        self._asset_name = asset_name
        self._filename = op.get_path(self.index, 'H', self.asset_name + c.filetype)
        self._data = self.get_historical_data()

        if fin_data:
            self._fin_data = self.save_financial_data()
            self._calculate_financials()
        else:
            self._fin_data = None


    def save_financial_data(self):
        """Function reading financial data from the database"""

        filename = op.get_path('SP500', 'F', self.asset_name + c.filetype)
        fin_data = pd.read_csv(filename, index_col = 0)

        return fin_data


    def get_financial_value(self, financial, iloc_value = -1):
        """Function return the financial value of the Ticker on a certain position"""

        value = self.fin_data.iloc[iloc_value][financial]
        if isinstance(value, str):
            if value == '-':
                return -1
            value = int(value.replace(',', ''))

        return value


    def _calculate_financials(self):
        """Method that calculates Financial values and stores in attributes."""

        #Calculate Price to Book Ratio
        equity = self.get_financial_value('Total Equity Gross Minority Interest')
        shares = self.get_financial_value('Ordinary Shares Number')
        bv_per_share = equity / shares
        self._pb_ratio = self.data.iloc[-1]['Close'] / bv_per_share

        #Calculate Return on Equity
        net_income = self.get_financial_value('Net Income Common Stockholders')
        equity = self.get_financial_value('Total Equity Gross Minority Interest')
        self._roe = 100 * (net_income / equity)

        #Calculate Price to Earnings Ratio
        eps_ = self.get_financial_value('Basic EPS')
        eps = max(0, eps_)
        self._pe = self.data.iloc[-1]['Close'] / eps

        return


class DAXTicker(Asset):
    """Class implementing the SP500 Assets"""

    index = 'DAX40'


    @property
    def filename(self):
        """Overrides filename property in Asset Class"""
        return self._filename


    @property
    def data(self):
        """Overrides data property in Asset Class"""
        return self._data


    @property
    def asset_name(self):
        """Asset Name property"""
        return self._asset_name


    def __init__(self, asset_name):

        self._asset_name = asset_name
        self._filename = op.get_path(self.index, 'H', self.asset_name + c.filetype)
        self._data = self.get_historical_data()
