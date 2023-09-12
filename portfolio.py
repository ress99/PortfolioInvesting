from sp500 import SP500
import numpy as np
import pandas as pd
from ticker import Ticker
import time
import config as c
import logging
import data_op as op
import datetime as dt

class Portfolio:

    #####
    #Normalizes the weights of all portfolio tickets
    #####
    def normalize_ticker_weights(self):

        logging.info('Normalizing the weights of the Portfolio.')
        weight_sum = sum(self.ticker_weights)
        logging.debug('The sum of the weights is %f', weight_sum)
        self.ticker_weights = [x / weight_sum for x in self.ticker_weights]
        self.ticker_weights = np.asarray(self.ticker_weights)
        logging.info('Portfolio weights normalized.')

        return
    
    #####
    #Checks if all ticker weights are positive
    #####
    def check_positivity(self):

        logging.info('Checking positivity of all weights of the Portfolio.')
        for tic_name in self.tickers_list:
            weight = self.prtf_dict[tic_name]['weight']
            logging.debug('Ticker %s has weight %f.', tic_name, weight)
            if weight < 0:
                logging.info('There are tickers with negative weights.')
                return False
        logging.info('All tickers have positive weight.')
        return True 

    #####
    #
    def apply_positivity(self, remove = True, normalize = True):
        
        logging.info('Applying positivity to the Portfolio weights.')
        if not self.check_positivity():
            for tic_name in self.tickers_list:
                weight = self.prtf_dict[tic_name]['weight']
                if weight < 0:
                    if remove:
                        logging.debug('Removing ticker %s with weight %d', tic_name, weight)
                        self.prtf_dict = [tic_name]
                    else:
                        new_value = np.array([positive_value for positive_value in self.ticker_weights if positive_value > 0]).mean()
                        self.change_ticker_weight(tic_name, new_value)
        if normalize:
            self.normalize_ticker_weights()


    def change_ticker_weight(self, tic_name, value):
        tic_name = tic_name.upper()
        logging.info('Changing weight of ticker %s to %f', tic_name, value)
        self._prtf_dict[tic_name]['weight'] = value


    def get_ticker_object(self, ticker_name):

        ticker_name = ticker_name.upper()
        if ticker_name in self.tickers_list:
            return self.prtf_dict[ticker_name]['ticker']
        else:
            logging.warning('The ticker %s is not in the Portfolio.', ticker_name)
            return

    def get_tickers_same_df(self, column = 'Close', start_date = None, end_date = None):

        lengths_list = [len(self.get_ticker_object(tic).data) for tic in self.tickers_list]
        max_length_index = lengths_list.index(max(lengths_list))
        max_lenght_tic = self.tickers_list[max_length_index]
        logging.debug('The ticker longest dataframe is the one of %s, with %d days.', max_lenght_tic, max(lengths_list))

        data = self.get_ticker_object(max_lenght_tic).data
        same_df_tickers = data['Date'].copy()
        same_df_tickers = same_df_tickers.to_frame()

        for tic in self.tickers_list:                                                         #For loop that passes all tickers
            
            logging.debug('Adding %s to the dataframe', tic)
            data = self.get_ticker_object(tic).data                                                                 #For following tickers
            same_df_tickers = same_df_tickers.merge(data[['Date', column]], on = 'Date', how = self.how_merge)       #Merge dataframes with outer or inner
            same_df_tickers.rename(columns={column: tic}, inplace = True)
        
        logging.info('All the %d tickers added to the same dataframe with %s merge.', len(self.tickers_list), self.how_merge)

        same_df_tickers = op.df_start_to_end_date(same_df_tickers, start_date = start_date, end_date = end_date)

        return same_df_tickers

    def add_ticker_to_dict(self, tic_name):

        logging.info('Adding %s to Portfolio.', tic_name)

        if tic_name in self.sp500.sp500_dict:
            if tic_name not in self.tickers_list:
                self.prtf_dict = [tic_name]
            else:
                logging.warning('Ticker %s is already in the Portfolio.', tic_name)                
        else:
            logging.warning('Ticker %s does not exist in the S&P500.', tic_name)
    
        return

    def remove_ticker_from_dict(self, tic_name):

        logging.info('Removing %s to Portfolio.', tic_name)

        if tic_name in self.sp500.sp500_dict:
            if tic_name in self.tickers_list:
                self.prtf_dict = [tic_name]
            else:
                logging.warning('Ticker %s is not in the Portfolio.', tic_name)                
        else:
            logging.warning('Ticker %s does not exist in the S&P500.', tic_name)
    
        return
    
    def substitute_tickers(self, out_ticker, in_ticker):

        out_ticker, in_ticker = out_ticker.upper(), in_ticker.upper()

        if out_ticker not in self.tickers_list:
            logging.warning('Ticker %s does not exist in the Portfolio.', out_ticker)
        elif in_ticker in self.tickers_list:
            logging.warning('Ticker %s already exists in the Portfolio.', in_ticker)
        elif in_ticker not in self.sp500.tickers_list:
            logging.warning('Ticker %s does not exist.', in_ticker)
            
        else:
            weight = self.prtf_dict[out_ticker]['weight']
            self.prtf_dict = [out_ticker, in_ticker]
            self.change_ticker_weight(in_ticker, weight)
            logging.info('Substituted %s for %s.', out_ticker, in_ticker)        



    def __init__(self, sp500, cardinality_constraint, column = 'Close', start_date = None, end_date = None, merge_option = 'inner'):

        logging.info('Initializing Portfolio object with %d stocks.', cardinality_constraint)

        self._prtf_dict = dict()
        self.sp500 = sp500
        self.cardinality_constraint = cardinality_constraint
        self.start_date = start_date
        self.end_date = end_date
        #TODO mudar data quando for inner
        self.merge_option = merge_option
        self.column = column

    @property
    def prtf_dict(self):
        return self._prtf_dict

    @prtf_dict.setter
    def prtf_dict(self, ticker_name):

        for tic in ticker_name:
            tic = tic.upper()

            if tic in self.tickers_list:                            #If it's already in the portfolio
                del self._prtf_dict[tic]
                logging.info('Ticker %s removed from Portfolio', tic)
            
            elif tic in self.sp500.tickers_list:                                        #If it exists in the sp500
                if self.sp500.sp500_dict[tic] is None:                                  #If not saved in the sp500 class
                    self.sp500.get_ticker(tic)                                          #Save it
                ticker_dict = dict()                                                    #Create new dictionary and save it under tic key
                ticker_dict = {'ticker': self.sp500.sp500_dict[tic], 'weight': -1}
                self._prtf_dict[tic] = ticker_dict
                logging.info('Ticker %s added to Portfolio', tic)
            
            else:
                logging.warning('Ticker %s does not exist in the S&P500', tic)

        return

    @property
    def tickers_list(self):
        return list(self.prtf_dict.keys())

    @property
    def nmbr_stocks(self):
        return len(self.prtf_dict)

    @property
    def ticker_weights(self):
        weights_list = []
        for ticker in self.prtf_dict:
            weight = self.prtf_dict[ticker]['weight']
            weights_list.append(weight)
        return weights_list

    @ticker_weights.setter
    def ticker_weights(self, new_weights):
        logging.info('Getting new ticker weights.')
        if len(new_weights) == self.nmbr_stocks:
            for idx, tic in enumerate(self.tickers_list):
                self.change_ticker_weight(tic, new_weights[idx])
        else:
            logging.warning('The new weights do not match the number of stocks in the Portfolio.')

    @property
    def prtf_df(self):
        
        lengths_list = [self.get_ticker_object(tic).total_days for tic in self.tickers_list]
        max_length_index = lengths_list.index(max(lengths_list))
        max_lenght_ticker = self.tickers_list[max_length_index]
        logging.debug('The longest dataframe is the one of %s, with %d days.', max_lenght_ticker, max(lengths_list))

        data = self.get_ticker_object(max_lenght_ticker).data
        same_df_tickers = data['Date'].copy()
        same_df_tickers = same_df_tickers.to_frame()

        for tic in self.tickers_list:                                                         #For loop that passes all tickers
            
            logging.debug('Adding %s to the dataframe', tic)
            data = self.get_ticker_object(tic).data                                                                 #For following tickers
            same_df_tickers = same_df_tickers.merge(data[['Date', self.column]], on = 'Date', how = self.merge_option)       #Merge dataframes with outer or inner
            same_df_tickers.rename(columns={self.column: tic}, inplace = True)
        
        logging.info('All the %d tickers added to the same dataframe with %s merge.', len(self.tickers_list), self.merge_option)

        same_df_tickers = op.df_start_to_end_date(same_df_tickers, start_date = self.start_date, end_date = self.end_date)

        return same_df_tickers

    @property
    def column(self):
        return self._column

    @column.setter
    def column(self, new_column):
        if new_column in op.quote_columns:
            self._column = new_column
        else:
            self._column = 'Close'

    @property
    def merge_option(self):
        return self._merge_option
    
    @merge_option.setter
    def merge_option(self, merge):

        how_merge_list = ['inner', 'outer']  

        if merge not in how_merge_list:
            logging.warning('The merge option %s is not valid. Will be set to default: \'inner\'', merge)
            self._merge_option = 'inner'
        else:
            self._merge_option = merge
            logging.info('The merge option is set to %s.', merge)
        return
    
    @property
    def start_date(self):
        return self._start_date
    
    @start_date.setter
    def start_date(self, date):

        logging.info('Setting the start date')
        self._start_date = op.date_str_to_dt(date)
        logging.info('Start date set as %s', self._start_date)
    
    @property
    def end_date(self):
        return self._end_date
    
    @end_date.setter
    def end_date(self, date):
    
        logging.info('Setting the end date')
        self._end_date = op.date_str_to_dt(date)
        logging.info('End date set as %s', self._end_date)
    
    ###Constraints
    def checks_cardinality(self):
        
        logging.info('Checking cardinality.')
        if self.nmbr_stocks > self.cardinality_constraint:
            logging.warning('There are %d stocks in the Portfolio, instead of the maximum of %d', self.nmbr_stocks, self.cardinality_constraint)
            return False
        
        logging.info('The Portfolio checks the cardinality constraint.')
        return True

    def apply_cardinality(self):

        logging.info('Applying cardinality constraint.')
        while not self.checks_cardinality():
            array_ticker_weights = np.array(self.ticker_weights)
            idx = np.argmin(array_ticker_weights)
            del self.prtf_dict[self.tickers_list[idx]]

    def apply_same_weights(self):

        logging.info('Applying same weights to the Portfolio.')
        new_weights = [1/self.nmbr_stocks] * self.nmbr_stocks
        logging.debug('The new weights are %s', str(new_weights))
        self.ticker_weights = new_weights
    
    def portfolio_return(self):
        
        logging.info('Calculating the Portfolio return from %s to %s.', self.start_date, self.end_date)
        self.apply_cardinality()
        self.apply_positivity()     
        total = 0   
        for idx, ticker_name in enumerate(self.tickers_list):
            ticker_object = self.get_ticker_object(ticker_name)
            stock_return = ticker_object.stock_return(column = self.column, start_date = self.start_date, end_date = self.end_date)
            total += stock_return * self.ticker_weights[idx]
        
        logging.info('The Portfolio return is %f', total)
        return total
    
    def portfolio_variance(self):
        weights = np.array(self.ticker_weights)
        covariance_matrix = self.prtf_df[self.tickers_list].cov()
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        return portfolio_variance
