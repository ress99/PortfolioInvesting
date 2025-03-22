"""Module with the Index Class and its Subclasses"""

import requests
import pandas as pd

import config as c
import data_op as op
from ticker import SPTicker, DAXTicker



class Index:
    """Class representing an Index"""

    def get_asset(self, asset_name):
        """Method that should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the 'get_asset' method.")


    def get_all_assets(self):
        """Method that should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the 'get_all_assets' method.")


    def create_dict(self):
        """Method that returns a Dictionary with all the Assets in the Index"""

        #Zip the asset list with None values that will be future asset objects
        zipped = zip(self.asset_list, [None] * len(self.asset_list))
        return dict(zipped)


    def __init__(self):

        self.asset_list = []
        self.asset_dict = {}


    def __len__(self):
        """Length of an Index returns the number of Assets in the Index."""
        return len(self.asset_list)


    def __getitem__(self, i):
        """Getter Method for asset Object in the dictionary."""

        if self.asset_dict[i] is None:
            self.get_asset(i)
        return self.asset_dict[i]


    def __iter__(self):
        """Iteration Implementation for asset list."""
        return iter(self.asset_list)


    def __str__(self):
        """String of the index returns the asset dictionary."""
        return str(self.asset_dict)


class SP500(Index):
    """Class implementing the SP500 Index"""

    name = 'SP500'
    name_asset_index = '^GSPC'


    def __init__(self):

        super().__init__()
        self.asset_list = self.get_all_assets()
        self.asset_dict = self.create_dict()


    def get_all_assets(self):
        """
        Implementation of get_all_assets Method.
        Reads all assets from index in database and returns the list.
        """

        path = op.get_path('SP500', filename = 'sp500tickers' + c.filetype)
        tickers_df = pd.read_csv(path)
        tickers = tickers_df['S&P500'].tolist()

        #Assets to remove from index
        to_remove = ['ABBV', 'ABNB', 'ALLE', 'AMCR', 'APTV', 'ANET', 'CZR',
                     'CARR', 'CTLT', 'CBOE','CDW', 'CDAY', 'CHTR', 'CFG',
                     'CEG', 'CTVA', 'FANG', 'DOW', 'ENPH', 'EPAM', 'ETSY', 
                     'FLT', 'FTV', 'FOXA', 'FOX', 'GEHC', 'GNRC', 'GM',
                     'HCA', 'HPE', 'HLT', 'HWM', 'HII', 'IR', 'INVH',
                     'IQV', 'KVUE', 'KEYS', 'KMI', 'KHC', 'LW', 'LYB',
                     'MPC', 'META', 'MRNA', 'NWSA', 'NWS', 'NCLH', 'NXPI',
                     'OGN', 'OTIS', 'PANW', 'PAYC', 'PYPL', 'PSX', 'QRVO', 
                     'NOW', 'SEDG', 'SYF', 'TRGP', 'TSLA', 'VICI', 'WRK',
                     'XYL', 'ZTS']       

        return [tic for tic in tickers if tic not in to_remove]


    def get_asset(self, asset_name, fin_data = False):
        """
        Implementation of get_asset method.
        Returns asset object from dictionary.
        """

        asset_name = asset_name.upper()

        #Check if asset exists in the index
        if asset_name in self.asset_list:

            #If the asset is not yet in asset_dict, initialize it
            if self.asset_dict[asset_name] is None:
                self.asset_dict[asset_name] = SPTicker(asset_name, fin_data)

            return self.asset_dict[asset_name]

        #If the asset not in the index
        else:
            print(f'Asset not found in {self.name}')
            return


    def download_asset_tickers(self):
        """Method that Retrieves and updates list of SP500 Assets."""

        agent = {"User-Agent":"Mozilla/5.0"}
        url = 'https://www.slickcharts.com/sp500'
        source=requests.get(url, headers=agent, timeout=10).text
        df_list = pd.read_html(source)

        sp500_df = df_list[0]
        sp500_df = sp500_df.sort_values('Company')
        sp500_list = sp500_df['Symbol'].tolist()
        sp500_list = [tic.replace('.', '-') for tic in sp500_list]

        sp500_series = pd.Series (sp500_list, name = 'S&P500')
        path = op.get_path(self.name, filename = self.name + c.filetype)
        sp500_series.to_csv(path, index = False)

        return


class DAX40(Index):
    """Class implementing the DAX40 Index"""

    name = 'DAX40'
    name_asset_index = '^GDAXI'


    def __init__(self):

        super().__init__()
        self.asset_list = self.get_all_assets()
        self.asset_dict = self.create_dict()


    def get_all_assets(self):
        """
        Implementation of get_all_assets Method.
        Reads all assets from index in database and returns the list.
        """

        path = op.get_path(self.name, filename = self.name + c.filetype)
        assets_df = pd.read_csv(path)
        assets = assets_df[self.name].tolist()

        #Assets to remove from index
        to_remove = ['CON.DE', 'P911.DE', 'SHL.DE', 'DTG.DE', 'ENR.DE', 
                     'BNR.DE', '1COV.DE', 'VNA.DE', 'ZAL.DE']

        return [tic for tic in assets if tic not in to_remove]


    def get_asset(self, asset_name):
        """
        Implementation of get_asset method.
        Returns asset object from dictionary.
        """

        asset_name = asset_name.upper()

        #Check if asset exists in the index
        if asset_name in self.asset_list:
            #If the asset is not yet in asset_dict, initialize it
            if self.asset_dict[asset_name] is None:
                self.asset_dict[asset_name] = DAXTicker(asset_name)

            return self.asset_dict[asset_name]

        #If the asset not in the index
        else:
            print(f'Asset not found in {self.name}')
            return


    def download_asset_tickers(self):
        """Method that Retrieves and updates list of DAX40 Assets."""

        url = "https://en.wikipedia.org/wiki/DAX"
        tables = pd.read_html(url)
        dax_table = tables[4]
        dax_companies = dax_table['Ticker'].dropna().tolist()

        dax40_series = pd.Series (dax_companies, name = self.name)
        path = op.get_path(self.name, filename = self.name + c.filetype)
        dax40_series.to_csv(path, index = False)

        return
