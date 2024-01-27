from ticker import Asset, SP_Ticker, DAX_Ticker
import data_op as op
import config as c
import pandas as pd

# from bs4 import BeautifulSoup
import requests

class Index:

    def get_asset(self):
        print('This method is not yet developed.') 
    
    #####
    #Creates a dictionary where all the Ticker objects
    #Can be saved and accessed
    #####
    def create_dict(self):

        #Zip the asset list with None values that are future asset objects
        zipped = zip(self.asset_list, [None] * len(self.asset_list))
        #Return the zip as a dictionary
        return dict(zipped)
    

    #####
    #Returns all the assets from the index
    #####
    def get_all_assets(self):

        return []

    def __init__(self):

        self.asset_list = []
        self.asset_dict = {}

    def __len__(self):
        return len(self.asset_list)

    def __getitem__(self, i):
        if self.asset_dict[i] is None:
            self.get_asset(i)
        return self.asset_dict[i]
    
    def __iter__(self):
        return iter(self.asset_list)
    
    def __str__(self):
        return str(self.asset_dict)

#####
#Class used to 
class SP500(Index):

    name = 'SP500'
    name_asset_index = '^GSPC'

    #####
    #Update list of all S&P500 tickers
    #####
    def download_asset_tickers(self):

        agent = {"User-Agent":"Mozilla/5.0"}
        url = 'https://www.slickcharts.com/sp500'
        source=requests.get(url, headers=agent).text
        df_list = pd.read_html(source)

        sp500_df = df_list[0]
        sp500_df = sp500_df.sort_values('Company')
        sp500_list = sp500_df['Symbol'].tolist()
        sp500_list = [tic.replace('.', '-') for tic in sp500_list]

        sp500_series = pd.Series (sp500_list, name = 'S&P500')
        path = op.get_path(self.name, filename = self.name + c.filetype)
        sp500_series.to_csv(path, index = False)

        return

    #####
    #Saves ticker inside the dictionary self.sp500_dict
    #####
    def get_asset(self, ticker_name, fin_data = False):

        ticker_name = ticker_name.upper()

        #If the ticker_name exists
        if ticker_name in self.asset_list:
            #If the ticker is not yet in self.sp500_dict, initialize it
            if self.asset_dict[ticker_name] is None:
                self.asset_dict[ticker_name] = SP_Ticker(ticker_name, fin_data)
            return self.asset_dict[ticker_name]

        #If the ticker_name is not valid
        else:
            print('Ticker name not found')
            return

    def get_all_assets(self):

        path = op.get_path('SP500', filename = 'sp500tickers' + c.filetype)
        tickers_df = pd.read_csv(path)
        tickers = tickers_df['S&P500'].tolist()
        to_remove = ['ABBV', 'ABNB', 'ALLE', 'AMCR', 'APTV', 'ANET', 'CZR', 'CARR', 'CTLT', 'CBOE', 
                     'CDW', 'CDAY', 'CHTR', 'CFG', 'CEG', 'CTVA', 'FANG', 'DOW', 'ENPH', 'EPAM', 'ETSY', 
                     'FLT', 'FTV', 'FOXA', 'FOX', 'GEHC', 'GNRC', 'GM', 'HCA', 'HPE', 'HLT', 'HWM', 'HII', 
                     'IR', 'INVH', 'IQV', 'KVUE', 'KEYS', 'KMI', 'KHC', 'LW', 'LYB', 'MPC', 'META', 'MRNA', 
                     'NWSA', 'NWS', 'NCLH', 'NXPI', 'OGN', 'OTIS', 'PANW', 'PAYC', 'PYPL', 'PSX', 'QRVO', 
                     'NOW', 'SEDG', 'SYF', 'TRGP', 'TSLA', 'VICI', 'WRK', 'XYL', 'ZTS']       

        return [tic for tic in tickers if tic not in to_remove]

    def __init__(self):

        self.asset_list = self.get_all_assets()
        self.asset_dict = self.create_dict()


class DAX40(Index):

    name = 'DAX40'
    name_asset_index = '^GDAXI'

    def download_asset_tickers(self):

        url = "https://en.wikipedia.org/wiki/DAX"
        tables = pd.read_html(url)
        dax_table = tables[4]
        dax_companies = dax_table['Ticker'].dropna().tolist()

        dax40_series = pd.Series (dax_companies, name = self.name)
        path = op.get_path(self.name, filename = self.name + c.filetype)
        dax40_series.to_csv(path, index = False)

        return

    def get_asset(self, asset_name):

        asset_name = asset_name.upper()

        #If the asset_name exists
        if asset_name in self.asset_list:
            #If the ticker is not yet in self.asset_dict, initialize it
            if self.asset_dict[asset_name] is None:
                self.asset_dict[asset_name] = DAX_Ticker(asset_name)
            return self.asset_dict[asset_name]

        #If the asset_name is not valid
        else:
            print('Ticker name not found')
            return

    def get_all_assets(self):

        path = op.get_path(self.name, filename = self.name + c.filetype)
        assets_df = pd.read_csv(path)
        assets = assets_df[self.name].tolist()

        to_remove = ['CON.DE', 'P911.DE', 'SHL.DE', 'DTG.DE', 'ENR.DE', 'BNR.DE', '1COV.DE', 'VNA.DE', 'ZAL.DE']
        return [tic for tic in assets if tic not in to_remove]

    def __init__(self):

        self.asset_list = self.get_all_assets()
        self.asset_dict = self.create_dict()