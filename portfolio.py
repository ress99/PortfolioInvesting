from sp500 import SP500
import numpy as np
import pandas as pd
from ticker import Ticker
import time


class Portfolio:

    def normalize_ticker_weights(self):

        weight_sum = sum(self.ticker_weights)
        self.ticker_weights = [x / weight_sum for x in self.ticker_weights]
        self.ticker_weights = np.asarray(self.ticker_weights)

        return
    
    def check_positivity(self):
        
        for ticker_name in self.tickers_list:
            if self.prtf_dict[ticker_name]['weight'] < 0:
                return False
        return True 

    def apply_positivity(self, remove = True, normalize = True):
        
        if not self.check_positivity():
            for ticker_name in self.tickers_list:
                if self.prtf_dict[ticker_name]['weight'] < 0:
                    if remove:
                        self.remove_ticker_from_dict(ticker_name)
                    else:
                        new_value = np.array([positive_value for positive_value in self.ticker_weights if positive_value > 0]).mean()
                        self.change_ticker_weight(ticker_name, new_value)
        if normalize:
            self.normalize_ticker_weights()



    def change_ticker_weight(self, ticker_name, value):
        ticker_name = ticker_name.upper()
        self.prtf_dict[ticker_name]['weight'] = value


    def get_ticker_object(self, ticker_name):

        ticker_name = ticker_name.upper()
        if ticker_name in self.tickers_list:
            return self.prtf_dict[ticker_name]['ticker']
        else:
            print('Ticker not in Portfolio.')
            return

    def get_tickers_same_df(self, column = 'Close', how_merge = 'outer'):

        how_merge_list = ['inner', 'outer']  

        if column not in Ticker.quote_columns:
            print ('Variable column is not valid')
            return

        if how_merge not in how_merge_list:
            print('Variable how_merge must be either \'outer\' or \'inner\'')
            return

        lengths_list = [len(self.get_ticker_object(tic).data) for tic in self.tickers_list]
        max_length_index = lengths_list.index(max(lengths_list))

        data = self.get_ticker_object(self.tickers_list[max_length_index]).data
        same_df_tickers = data['Date'].copy()
        same_df_tickers = same_df_tickers.to_frame()

        for tic in self.tickers_list:                                                         #For loop that passes all tickers

            data = self.get_ticker_object(tic).data                                                                 #For following tickers
            same_df_tickers = same_df_tickers.merge(data[['Date', column]], on = 'Date', how = how_merge)       #Merge dataframes with outer or inner
            same_df_tickers.rename(columns={column: tic}, inplace = True)
        
        return same_df_tickers

    def add_ticker_to_dict(self, ticker_name):

        if ticker_name not in self.tickers_list:
            self.prtf_dict = [ticker_name]
        else:
            print('Not able to add this ticker to Portfolio. Check if it exists or is already in the Portfolio.')
        return

    def remove_ticker_from_dict(self, ticker_name):

        if ticker_name in self.tickers_list:
            self.prtf_dict = [ticker_name]
        else:
            print('Not able to remove this ticker from Portfolio. Check if it exists or if it is in the Portfolio.')
        return


    def __init__(self, sp500, cardinality_constraint):
        self._prtf_dict = dict()
        self.sp500 = sp500
        self.cardinality_constraint = cardinality_constraint
        self._ticker_weights = []



    @property
    def prtf_dict(self):
        return self._prtf_dict



    @prtf_dict.setter
    def prtf_dict(self, ticker_name):

        for tic in ticker_name:
            tic = tic.upper()

            if tic in self.tickers_list:                            #If it's already in the portfolio
                del self.prtf_dict[tic]
            
            elif tic in self.sp500.tickers_list:                                        #If it exists
                if self.sp500.sp500_dict[tic] is None:                                  #If not saved in the sp500 class
                    self.sp500.get_ticker(tic)                                               #Save it
                ticker_dict = dict()                                                    #Create new dictionary and save it under tic key
                ticker_dict = {'ticker': self.sp500.sp500_dict[tic], 'weight': -1}
                self.prtf_dict[tic] = ticker_dict
            
            else:
                print('Ticker', tic, 'does not exist.')

        return

    @property
    def tickers_list(self):
        return list(self.prtf_dict.keys())

    @property
    def nmbr_stocks(self):
        return len(self.prtf_dict)

    @property
    def ticker_weights(self):
        return self._ticker_weights

    @ticker_weights.getter
    def ticker_weights(self):
        weights_list = []
        for ticker in self.prtf_dict:
            weight = self.prtf_dict[ticker]['weight']
            weights_list.append(weight)
        return weights_list

    @ticker_weights.setter
    def ticker_weights(self, new_weights):
        if len(new_weights) == len(self.tickers_list):
            for idx, tic in enumerate(self.tickers_list):
                self.change_ticker_weight(tic, new_weights[idx])
        else:
            print('The lenght of the new weights must be similar to the tickers_list')

    
    ###Constraints
    def checks_cardinality(self):
        
        if self.nmbr_stocks > self.cardinality_constraint:
            return False
        
        return True

    def apply_cardinality(self):

        while not self.checks_cardinality():
            array_ticker_weights = np.array(self.ticker_weights)
            idx = np.argmin(array_ticker_weights)
            print('Ticker', self.tickers_list[idx], 'will be removed')
            del self.prtf_dict[self.tickers_list[idx]]

    def apply_same_weights(self):

        new_weights = [1/len(self.tickers_list)] * len(self.tickers_list)
        self.ticker_weights = new_weights
    
    def portfolio_return(self, column = 'Close', start_date = None, end_date = None):
        
        self.apply_cardinality()
        self.apply_positivity()     
        total = 0
        for idx, ticker_name in enumerate(self.tickers_list):
            tic = self.get_ticker_object(ticker_name)
            stock_return = tic.stock_return(column = column, start_date = start_date, end_date = end_date)
            total += stock_return * self.ticker_weights[idx]
            #print(stock_return, self.ticker_weights[idx], stock_return * self.ticker_weights[idx])
        
        return total

    


            
    




