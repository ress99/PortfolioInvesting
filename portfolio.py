import numpy as np
import pandas as pd
import time
import config as c
import logging
import data_op as op
import datetime as dt
import json
import os

class Portfolio:

    #####
    #Normalizes the weights of all portfolio tickets
    #####
    def normalize_asset_weights(self):

        logging.info('Normalizing the weights of the Portfolio.')
        weight_sum = sum(self.asset_weights)
        logging.debug('The sum of the weights is %f', weight_sum)
        self.asset_weights = [x / weight_sum for x in self.asset_weights]
        self.asset_weights = np.asarray(self.asset_weights)
        logging.info('Portfolio weights normalized.')

        return
    
    #####
    #Checks if all ticker weights are positive
    #####
    def check_positivity(self):

        logging.info('Checking positivity of all weights of the Portfolio.')
        for asset_name in self:
            weight = self[asset_name]['weight']
            logging.debug('Asset %s has weight %f.', asset_name, weight)
            if weight < 0:
                logging.info('There are assets with negative weights.')
                return False
        logging.info('All assets have positive weight.')
        return True 

    #####
    #
    def apply_positivity(self, remove = True, normalize = True):
        
        logging.info('Applying positivity to the Portfolio weights.')
        if not self.check_positivity():
            for asset_name in self:
                weight = self[asset_name]['weight']
                if weight < 0:
                    if remove:
                        logging.debug('Removing asset %s with weight %d', asset_name, weight)
                        self.prtf_dict = [asset_name]
                    else:
                        new_value = np.array([positive_value for positive_value in self.asset_weights if positive_value > 0]).mean()
                        self.change_asset_weight(asset_name, new_value)
        if normalize:
            self.normalize_asset_weights()


    def change_asset_weight(self, asset_name, value):
        asset_name = asset_name.upper()
        logging.info('Changing weight of ticker %s to %f', asset_name, value)
        self._prtf_dict[asset_name]['weight'] = value


    def get_asset_object(self, asset_name):

        asset_name = asset_name.upper()
        if asset_name in self:
            return self[asset_name]['object']
        else:
            logging.warning('The asset %s is not in the Portfolio.', asset_name)
            return

    def get_assets_same_df(self, column = 'Close', start_date = None, end_date = None):

        lengths_list = [len(self.get_asset_object(asset).data) for asset in self]
        longest_index = lengths_list.index(max(lengths_list))
        longest_asset = self.asset_list[longest_index]
        logging.debug('The longest asset is %s, with %d days.', longest_asset, max(lengths_list))

        data = self.get_asset_object(longest_asset).data
        df = data['Date'].copy()
        df = df.to_frame()

        for asset in self:                                                         #For loop that passes all tickers
            
            logging.debug('Adding %s to the dataframe', asset)
            data = self.get_asset_object(asset).data                                                                 #For following tickers
            df = df.merge(data[['Date', column]], on = 'Date', how = self.merge_option)       #Merge dataframes with outer or inner
            df.rename(columns={column: asset}, inplace = True)
        
        logging.info('All the %d assets added to the same dataframe with %s merge.', len(self), self.merge_option)

        df = op.df_start_to_end_date(df, start_date = start_date, end_date = end_date)
        #df.set_index('Date', inplace=True)

        return df

    def get_asset_index(self, asset):

        if asset not in self.all_assets:
            logging.warning(('Asset %s not present in any of the indexes', asset))
            return
        
        for i in self.indexes.values():
            if asset in i:
                return i
        

    def add(self, asset):

        logging.info('Adding %s to Portfolio.', asset)

        if asset in self.all_assets:
            if asset not in self:
                self.prtf_dict = [asset]
            else:
                logging.warning('Asset %s is already in the Portfolio.', asset)                
        else:
            logging.warning('Asset %s does not exist.', asset)
    
        return

    def remove(self, asset):

        logging.info('Removing %s from Portfolio.', asset)

        if asset in self.all_assets:
            if asset in self:
                self.prtf_dict = [asset]
            else:
                logging.warning('Asset %s is not in the Portfolio.', asset)                
        else:
            logging.warning('Asset %s does not exist.', asset)
    
        return
    
    def substitute_assets(self, out_asset, in_asset):

        out_asset, in_asset = out_asset.upper(), in_asset.upper()

        if out_asset not in self.asset_list:
            logging.warning('Asset %s does not exist in the Portfolio.', out_asset)
        elif in_asset in self.asset_list:
            logging.warning('Asset %s already exists in the Portfolio.', in_asset)
        elif in_asset not in self.all_assets:
            logging.warning('Asset %s does not exist.', in_asset)
            
        else:
            weight = self.prtf_dict[out_asset]['weight']
            self.prtf_dict = [out_asset, in_asset]
            self.change_asset_weight(in_asset, weight)
            logging.info('Substituted %s for %s.', out_asset, in_asset)        


    def get_index(self, asset):

        for i in self.indexes:
            if asset in self.indexes[i]:
                return self.indexes[i]
        return None


    def __init__(self, indexes, cardinality_constraint = None, column = 'Close', start_date = None, end_date = None, merge_option = 'inner', filename = None):

        logging.info('Initializing Portfolio object with %d stocks.', cardinality_constraint)

        self._prtf_dict = dict()
        self.indexes = indexes

        if filename is not None:
            self.init_from_file(filename)
        
        else:
            self.cardinality_constraint = cardinality_constraint
            self.start_date = start_date
            self.end_date = end_date
            #TODO mudar data quando for inner
            self.merge_option = merge_option
            self.column = column

    def __len__(self):
        return len(self._prtf_dict)

    def __getitem__(self, i):
        return self._prtf_dict[i]
    
    def __iter__(self):
        return iter(self._prtf_dict)
    
    def __str__(self):
        return str(self._prtf_dict)
    
    '''def __setitem__(self, key, value):
        ticker_dict = {'object': key, 'weight': value}
        self._prtf_dict[key] = ticker_dict'''

    def __delitem__(self, key):
        del self._prtf_dict[key]

    @property
    def prtf_dict(self):
        return self._prtf_dict

    @prtf_dict.setter
    def prtf_dict(self, asset_name):

        for asset in asset_name:
            asset = asset.upper()

            if asset in self:                            #If it's already in the portfolio
                del self._prtf_dict[asset]
                logging.info('Asset %s removed from Portfolio', asset)
            
            elif asset in self.all_assets:                                        #If it exists in the sp500
                index = self.get_asset_index(asset)
                if index[asset] is None:                                  #If not saved in the sp500 class
                    index.get_asset(asset)                                          #Save it
                ticker_dict = dict()                                                    #Create new dictionary and save it under tic key
                ticker_dict = {'object': index[asset], 'weight': -1}
                self._prtf_dict[asset] = ticker_dict
                logging.info('Asset %s added to Portfolio', asset)
            
            else:
                logging.warning('Asset %s does not exist in the S&P500', asset)

        return

    @property
    def indexes(self):
        return self._indexes
    
    @indexes.setter
    def indexes(self, idxs):
        self._indexes = {i.name: i for i in idxs}

    @property
    def all_assets(self):
        assets = []
        aux_list = [list(i) for i in self.indexes.values()]
        [assets.extend(i) for i in aux_list]
        return assets

    @property
    def asset_list(self):
        return list(self.prtf_dict.keys())

    @property
    def nmbr_stocks(self):
        return len(self.prtf_dict)

    @property
    def asset_weights(self):
        weights_list = []
        for ticker in self.prtf_dict:
            weight = self.prtf_dict[ticker]['weight']
            weights_list.append(weight)
        return weights_list

    @asset_weights.setter
    def asset_weights(self, new_weights):
        logging.info('Getting new ticker weights.')
        if len(new_weights) == self.nmbr_stocks:
            for idx, tic in enumerate(self.asset_list):
                self.change_asset_weight(tic, new_weights[idx])
        else:
            logging.warning('The new weights do not match the number of stocks in the Portfolio.')

    @property
    def prtf_df(self):
        
        return self.get_assets_same_df(column = 'Close', 
                                       start_date = self.start_date, 
                                       end_date=self.end_date)
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
    
    def prtf_to_dict(self):

        return_dict = {'asset_list': self.asset_list,
                       'asset_weights': self.asset_weights,
                       'indexes': list(self.indexes.keys()), 
                       'cardinality_constraint': self.cardinality_constraint,
                       'start_date': self.start_date, 
                       'end_date': self.end_date,
                       'merge_option': self.merge_option,
                       'column': self.column}
        
        return return_dict

    def write_json(self, name = None):

        if name is None:
            now = dt.datetime.now()
            name = now.strftime("%Y-%m-%d_%H-%M")

        write_dict = self.prtf_to_dict()
        
        with open(os.path.join(c.prtf_folder, name + '.json'), 'w') as file:
            json.dump(write_dict, file)

    def init_from_file(self, filename = None):

        with open(os.path.join(c.prtf_folder, filename), 'r') as file:
            content = json.load(file)
        
        self.prtf_dict = content['asset_list']
        self.asset_weights = content['asset_weights']
        self.cardinality_constraint = content['cardinality_constraint']
        self.start_date = content['start_date']
        self.end_date = content['end_date']
        self.merge_option = content['merge_option']
        self.column = content['column']

        return


    ###Constraints
    def checks_cardinality(self):
        
        logging.info('Checking cardinality.')
        if self.cardinality_constraint is None:
            logging.info('The Portfolio checks the cardinality constraint.')
            return True
        if self.nmbr_stocks > self.cardinality_constraint:
            logging.warning('There are %d stocks in the Portfolio, instead of the maximum of %d', self.nmbr_stocks, self.cardinality_constraint)
            return False
        
        logging.info('The Portfolio checks the cardinality constraint.')
        return True

    def apply_cardinality(self):

        logging.info('Applying cardinality constraint.')
        while not self.checks_cardinality():
            array_asset_weights = np.array(self.asset_weights)
            idx = np.argmin(array_asset_weights)
            del self.prtf_dict[self.asset_list[idx]]

    def apply_same_weights(self):

        logging.info('Applying same weights to the Portfolio.')
        new_weights = [1/self.nmbr_stocks] * self.nmbr_stocks
        logging.debug('The new weights are %s', str(new_weights))
        self.asset_weights = new_weights
    
    def portfolio_return(self):
        
        logging.info('Calculating the Portfolio return from %s to %s.', 
                     self.prtf_df['Date'].iloc[0], 
                     self.prtf_df['Date'].iloc[-1])

        cumulative_returns = (self.prtf_df[self].pct_change() + 1).cumprod() - 1
        prtf_return = np.dot(cumulative_returns.iloc[-1], self.asset_weights) * 100

        return prtf_return
    
       
    def portfolio_variance(self):
        asset_weights_array = np.array(self.asset_weights)
        covariance_matrix = self.prtf_df[self].cov()
        portfolio_variance = asset_weights_array.T.dot(covariance_matrix).dot(asset_weights_array)
        #portfolio_risk = np.sqrt(portfolio_variance)
        return portfolio_variance

    
    def portfolio_pe(self):

        pe = 0
        for i in self:
            pe += self[i]['object'].pe
        return pe / 10
    
    def portfolio_roe(self):

        roe = 0
        for i in self:
            roe += self[i]['object'].roe
        return roe / 10