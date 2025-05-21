"""Module with the implementation of the Portfolio Class"""

import datetime as dt
import json
import os
import logging
from cachetools import LRUCache
import numpy as np

import config as c
from fitness import Fitness
import data_op as op



logger = logging.getLogger(__name__)

class Portfolio:
    """
    Implementation of the Portfolio Class.
    Attributes and Methods that can be used by the user dealing with the Portfolio. 
    """

    cache = LRUCache(maxsize=c.size_cache)
    count = 0


    def __init__(self, indexes, cardinality_constraint = None, column = 'Close',
                 start_date = None, end_date = None, merge_option = 'inner', filename = None):

        logger.info('Initializing Portfolio object with %d stocks.', cardinality_constraint)
        self._prtf_dict = dict()
        self.indexes = indexes

        if filename is not None:
            self.init_from_file(filename)

        else:
            self.cardinality_constraint = cardinality_constraint
            self.start_date = start_date
            self.end_date = end_date
            self.merge_option = merge_option
            self.column = column
            self.invalid = False
            self.penalty = 0

        self.old_key = None


    def __len__(self):
        """Sets length of portfolio object as the number of assets."""
        return len(self._prtf_dict)


    def __getitem__(self, i):
        """When trying to get an item from the portfolio object, returns the asset object."""
        return self._prtf_dict[i]


    def __iter__(self):
        """Sets the iteration object as the self._prtf_dict"""
        return iter(self._prtf_dict)


    def __str__(self):
        """String of the object returns the self._prtf_dict"""
        return str(self._prtf_dict)


    def __delitem__(self, key):
        """del action removes the asset from the self._prtf_dict"""
        del self._prtf_dict[key]


    @property
    def fitness(self):
        """Property for fitness attribute."""
        return self._fitness


    @fitness.setter
    def fitness(self, objectives):
        """
        Fitness setter method.
        fitness attribute is a Fitness object with the objectives as input
        """
        if isinstance(objectives, tuple):
            self._fitness = Fitness(objectives)
        else:
            print("The variable is not a tuple.")


    @property
    def prtf_dict(self):
        """
        Property of the portfolio dictionary.
        Returns the private _prtf_dict,
        """
        return self._prtf_dict

    @prtf_dict.setter
    def prtf_dict(self, asset_names):
        """
        Setter method for prtf_dict.
        Changes private method _prtf_dict to ensure encapsulation.
        Assets are added to portfolio (if not yet in it) or removed (if already in portfolio).
        Takes as input a list of asset names.
        """

        #Goes through all assets in the list
        for asset in asset_names:
            asset = asset.upper()

            #Checks if asset is already in the portfolio  - if so removes it
            if asset in self:
                del self._prtf_dict[asset]
                logger.info('Asset %s removed from Portfolio', asset)

            #If not in the portfolio but exists in the indexes, adds it to the portfolio
            elif asset in self.all_assets:
                index = self.get_asset_index(asset)
                asset_dict = dict()
                asset_dict = {'object': index[asset], 'weight': -1}
                self._prtf_dict[asset] = asset_dict
                logger.info('Asset %s added to Portfolio', asset)

            #If the asset does not exist, issues a warning
            else:
                logger.warning('Asset %s does not exist in the index list', asset)

        return


    @property
    def indexes(self):
        """Property for the dictionary of indexes."""
        return self._indexes


    @indexes.setter
    def indexes(self, idxs):
        """
        Setter method for indexes.
        Uses the private attribute _indexes.
        """
        self._indexes = {i.name: i for i in idxs}

    @property
    def index_objects(self):
        """Property for the index_objects attribute."""
        return list(self._indexes.values())

    @property
    def all_assets(self):
        """
        Property for the all_assets attribute.
        Returns all valid assets inside the indexes.
        """
        assets = []
        for index_assets in self.indexes.values():
            assets.extend(index_assets)
        return assets

    @property
    def asset_list(self):
        """
        Property for the asset_list attribute.
        Returns the list of assets in the portfolio.
        """
        return list(self._prtf_dict.keys())

    @property
    def nmbr_assets(self):
        """Property for the nmbr_assets in the portfolio."""
        return len(self._prtf_dict)

    @property
    def asset_weights(self):
        """
        Property for the asset weights.
        Returns the weights of all assets in the portfolio.
        """

        #Iterates over all assets of the portfolio and makes a list with it's assets
        return [asset_dict['weight'] for asset_dict in list(self._prtf_dict.values())]


    @asset_weights.setter
    def asset_weights(self, new_weights):
        """
        Setter method for the asset_weights attribute.
        Takes as input a list of the new weights.
        If the list has a valid size, the asset weights are changed
        """

        logger.info('Getting new asset weights.')
        if len(new_weights) == self.nmbr_assets:
            for idx, tic in enumerate(self.asset_list):
                self.change_asset_weight(tic, new_weights[idx])
        else:
            logger.warning('The new weights do not match the number of stocks in the Portfolio.')


    @property
    def prtf_df(self):
        """Property for the DataFrame of the portfolio.
        Uses cache to store attribute for easier future access."""

        #Gets the key for the cache
        key = tuple(self.settings)

        #If the key is not in the cache, calculate the DataFrame and store it in cache
        if key not in Portfolio.cache:
            dataframe = self.set_assets_same_df(column = 'Close',
                                        start_date = self.start_date,
                                        end_date=self.end_date)
            Portfolio.cache[key] = dataframe

        #Return the value stored in the cache
        return Portfolio.cache[key]


    @property
    def daily_returns(self):
        """
        Property for the daily returns of the assets in the portfolio.
        Returns a DataFrame.
        """

        return self.prtf_df[self].pct_change()


    @property
    def settings(self):
        """
        Property for the settings of the portfolio.
        Returns a list.
        Used for caching purposes.
        """
        return [tuple(sorted(self.asset_list)), 'Close', self.start_date, self.end_date]


    @property
    def column(self):
        """Property for accessing the private _column attribute."""
        return self._column


    @column.setter
    def column(self, new_column):
        """Setter method for the private _column attribute."""

        if new_column in op.quote_columns:
            self._column = new_column
        else:
            self._column = 'Close'

    @property
    def merge_option(self):
        """Property for accessing the private _merge_option attribute."""
        return self._merge_option


    @merge_option.setter
    def merge_option(self, merge):
        """Setter method for the private _merge_option attribute."""

        how_merge_list = ['inner', 'outer']

        if merge not in how_merge_list:
            logger.warning('%s merge is not valid. Will be set to default: \'inner\'', merge)
            self._merge_option = 'inner'
        else:
            self._merge_option = merge
            logger.info('The merge option is set to %s.', merge)

        return


    @property
    def start_date(self):
        """Property for accessing the private _start_date attribute."""
        return self._start_date


    @start_date.setter
    def start_date(self, date):
        """Setter method for the private _start_date attribute."""

        logger.info('Setting the start date')
        self._start_date = date
        logger.info('Start date set as %s', self._start_date)


    @property
    def end_date(self):
        """Property for accessing the private _end_date attribute."""
        return self._end_date


    @end_date.setter
    def end_date(self, date):
        """Setter method for the private _end_date attribute."""

        logger.info('Setting the end date')
        self._end_date = date
        logger.info('End date set as %s', self._end_date)


    @property
    def total_portfolio_return(self):
        """Property calculating the total portfolio return."""

        return np.round(self.get_portfolio_returns_df().iloc[-1], 2)


    @property
    def annualized_portfolio_return(self):
        """Property calculating the annualized portfolio return."""

        #First method implemented
        # returns_df = self.get_portfolio_returns_df()
        # total_days = (returns_df.index[-1] - returns_df.index[0]).days
        # total_years = total_days / 365.25
        # annualized_return = (1 + self.total_portfolio_return / 100) ** (1 / total_years) - 1

        #Calculate cumulative returns
        cum_returns = (self.daily_returns + 1).cumprod() - 1
        #Adjust cumulative returns to the corresponding asset weights
        total_cum_returns = np.dot(cum_returns.iloc[-1], self.asset_weights)
        #Annualize the cummulative returns and annualize the number
        annualized_return = (total_cum_returns + 1) ** (1 / (self.prtf_df.shape[0] / 252)) - 1

        return np.round(annualized_return * 100, 2)


    def normalize_asset_weights(self):
        """Normalizes the weights of all portfolio assets."""

        logger.info('Normalizing the weights of the Portfolio.')
        weight_sum = sum(self.asset_weights)
        logger.debug('The sum of the weights is %f', weight_sum)
        self.asset_weights = [x / weight_sum for x in self.asset_weights]
        self.asset_weights = np.asarray(self.asset_weights)
        logger.info('Portfolio weights normalized.')

        return


    def adjust_decimals(self, decimals = 4):
        """Adjusts decimals of asset weights."""

        rounded = [round(num, decimals) for num in self.asset_weights]
        diff = round(1 - sum(rounded), decimals)
        max_value = max(rounded)
        idx = rounded.index(max_value)
        rounded[idx] = round(rounded[idx] + diff, decimals)
        self.asset_weights = np.asarray(rounded)

        return


    def check_positivity(self):
        """
        Checks positivity constraint (all assets are positive).
        Returns True if all are positive; False otherwise.
        """

        logger.info('Checking positivity of all weights of the Portfolio.')
        for asset_name in self:
            weight = self[asset_name]['weight']
            logger.debug('Asset %s has weight %f.', asset_name, weight)
            if weight < 0:
                logger.info('There are assets with negative weights.')
                return False
        logger.info('All assets have positive weight.')
        return True


    def apply_positivity(self, remove = True, normalize = True):
        """
        Applies positivity constraint to asset weights.
        If remove is True, assets with negative weight are removed.
        If normalize is True, the asset weights are normalized in the end.
        """

        logger.info('Applying positivity to the Portfolio weights.')
        if not self.check_positivity():
            for asset_name in self:
                weight = self[asset_name]['weight']
                if weight < 0:
                    if remove:
                        logger.debug('Removing asset %s with weight %d', asset_name, weight)
                        self.prtf_dict = [asset_name]
                    else:
                        new_value = np.array(
                            [positive_value for positive_value
                             in self.asset_weights if positive_value > 0]).mean()
                        self.change_asset_weight(asset_name, new_value)
        if normalize:
            self.normalize_asset_weights()


    def change_asset_weight(self, asset_name, value):
        """
        Changes asset weight in the prtf_dict.
        Takes as input the asset name and the new weight value.
        """

        logger.debug('Changing weight of asset %s to %f', asset_name, value)
        asset_name = asset_name.upper()
        value_ = round(value, 4)
        self._prtf_dict[asset_name]['weight'] = value_


    def get_object(self, asset_name):
        """Returns the object on the dictionary from an asset.
        Gets as input the asset_name"""

        asset_name = asset_name.upper()
        if asset_name in self:
            return self[asset_name]['object']
        else:
            logger.warning('The asset %s is not in the Portfolio.', asset_name)
            return


    def get_index_portfolio(self, indexes, start_date = None, end_date = None):
        """
        Creates a portfolio with all the index assets.
        Takes as input a list of indexes and a start and end date.
        """

        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        assets = [i.name_asset_index for i in indexes]
        prtf = Portfolio(
            indexes,
            cardinality_constraint = len(indexes),
            start_date = start_date,
            end_date = end_date)

        prtf.prtf_dict = assets
        prtf.apply_same_weights()

        return prtf


    def set_assets_same_df(self, column = 'Close', start_date = None, end_date = None):
        """
        Merges all the portfolio assets into the same DataFrame.
        column variable indicates the column from each asset to use.
        self.merge_option variable indicates the merge process.
        """

        lengths_list = [len(self.get_object(asset).data) for asset in self]
        longest_index = lengths_list.index(max(lengths_list))
        longest_asset = self.asset_list[longest_index]
        logger.debug('The longest asset is %s, with %d days.', longest_asset, max(lengths_list))

        data = self.get_object(longest_asset).data
        df = data['Date'].copy()
        df = df.to_frame()

        #For loop through all assets
        for asset in self:

            logger.debug('Adding %s to the dataframe', asset)
            data = self.get_object(asset).data
            df = df.merge(data[['Date', column]], on = 'Date', how = self.merge_option)
            df.rename(columns={column: asset}, inplace = True)

        logger.info('All the %d assets added to the same dataframe with %s merge.',
                    len(self), self.merge_option)

        df = op.df_start_to_end_date(df, start_date = start_date, end_date = end_date)

        return df


    def get_asset_index(self, asset):
        """
        Get the the index of a given asset.
        Takes as input the asset name.
        """

        if asset not in self.all_assets:
            logger.warning(('Asset %s not present in any of the indexes', asset))

        for i in self.indexes.values():
            if asset in i:
                return i
        return


    def get_index(self, asset):
        """Gets index of an asset."""

        for i in self.indexes:
            if asset in self.indexes[i]:
                return self.indexes[i]
        return None


    def add(self, asset):
        """Add asset to the prtf_dict"""

        logger.info('Adding %s to Portfolio.', asset)

        if asset in self.all_assets:
            if asset not in self:
                self.prtf_dict = [asset]
            else:
                logger.warning('Asset %s is already in the Portfolio.', asset)
        else:
            logger.warning('Asset %s does not exist.', asset)

        return


    def remove(self, asset):
        """Remove asset from the prtf_dict"""

        logger.info('Removing %s from Portfolio.', asset)

        if asset in self.all_assets:
            if asset in self:
                self.prtf_dict = [asset]
            else:
                logger.warning('Asset %s is not in the Portfolio.', asset)
        else:
            logger.warning('Asset %s does not exist.', asset)

        return


    def swap_assets(self, out_asset, in_asset):
        """
        Swap assets in the portfolio.
        Makes sure the swap of assets is valid.
        """

        out_asset, in_asset = out_asset.upper(), in_asset.upper()

        if out_asset not in self.asset_list:
            logger.info('Asset %s does not exist in the Portfolio.', out_asset)
        elif in_asset in self.asset_list:
            logger.info('Asset %s already exists in the Portfolio.', in_asset)
        elif in_asset not in self.all_assets:
            logger.info('Asset %s does not exist.', in_asset)

        else:
            weight = self.prtf_dict[out_asset]['weight']
            self.prtf_dict = [out_asset, in_asset]
            self.change_asset_weight(in_asset, weight)
            logger.info('Substituted %s for %s in the portfolio.', out_asset, in_asset)
            return True

        return False


    def apply_penalty(self):
        """Applies penalty to the portfolio evaluation"""

        multiplier = 1 + self.penalty
        self.fitness.values = self.fitness.values * multiplier
        self.penalty = 0

        return


    def obj_to_dict(self):
        """Method stores class settings as a dictionary."""

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
        """
        Store object settings in a json file.
        Takes a name as input; if none is provided, the current datetime is used.
        """

        if name is None:
            now = dt.datetime.now()
            name = now.strftime("%Y-%m-%d_%H-%M")

        write_dict = self.obj_to_dict()

        with open(os.path.join(c.prtf_folder, name + '.json'), 'w', encoding="utf-8") as file:
            json.dump(write_dict, file)


    def init_from_file(self, filename = None):
        """
        Initialization of object from a json file.
        Called when a filename is provided in to the initialization function.
        """

        with open(os.path.join(c.prtf_folder, filename), 'r', encoding="utf-8") as file:
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
    def check_cardinality(self):
        """Method to check wether portfolio respects the cardinality constraint."""

        if self.cardinality_constraint is None:
            return True
        if self.nmbr_assets > self.cardinality_constraint:
            return False

        logger.info('The Portfolio checks the cardinality constraint.')
        return True


    def apply_cardinality(self):
        """
        Method applying the cardinality contraint.
        Removes the asset with the lowest weight until the constraint is met.
        """

        logger.info('Applying cardinality constraint.')
        while not self.check_cardinality():
            array_asset_weights = np.array(self.asset_weights)
            idx = np.argmin(array_asset_weights)
            del self.prtf_dict[self.asset_list[idx]]


    def apply_same_weights(self):
        """Method that applied the same weights to all assets."""

        logger.info('Applying same weights to the Portfolio.')
        new_weights = [1/self.nmbr_assets] * self.nmbr_assets
        logger.debug('The new weights are %s', str(new_weights))
        self.asset_weights = new_weights
        self.adjust_weights(self.asset_weights)


    def adjust_weights(self, new_weights):
        """Method to adjust the asset weights."""

        sum_ = sum(new_weights)
        if sum_ < 1:
            nmbr = (1 - sum_) // 0.0001
            self.asset_weights = [
                value + 0.0001 if idx < nmbr else value for idx, value in enumerate(new_weights)
                ]
        elif sum_ >= 1:
            nmbr = (sum_ - 1) // 0.0001
            self.asset_weights = [
                value - 0.0001 if idx < nmbr else value for idx, value in enumerate(new_weights)
                ]


    def get_portfolio_returns_df(self, base = 100):
        """
        Calculates the portfolio return from start to end date.
        Returns a dataframe.
        """

        df = self.prtf_df.copy()
        df.set_index('Date', inplace=True)

        #Divide the prtf_df by the first value and multiply by the base (100 by default)
        df = (df / df.iloc[0] - 1) * base

        #Adjust the returns by the asset weights
        return df.dot(self.asset_weights)


    def portfolio_risk(self, annualized = True):
        """Method calculating the portfolio risk"""

        asset_weights_array = np.array(self.asset_weights)
        returns_df = self.daily_returns.dropna()
        covariance_matrix = returns_df.cov()
        prtf_risk = np.sqrt(asset_weights_array.T.dot(covariance_matrix).dot(asset_weights_array))
        if annualized:
            return prtf_risk * np.sqrt(252)
        return prtf_risk


    def sharpe_ratio(self, risk_free_rate = 0.03):
        """Method calculating the Sharpe Ratio"""

        # portfolio_returns = self.daily_returns.dot(self.asset_weights)
        # annualized_return = portfolio_returns.mean() * 252
        # portfolio_std_dev = portfolio_returns.std() * np.sqrt(252)
        # sharpe_ratio = (annualized_return - risk_free_rate) / portfolio_std_dev

        annualized_return = self.annualized_portfolio_return() / 100
        portfolio_daily_returns = self.daily_returns.dot(self.asset_weights)
        annualized_portfolio_std = portfolio_daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_portfolio_std

        return sharpe_ratio


    def value_at_risk(self, confidence = 0.95):
        """Method calculating the Value at Risk (VaR)"""

        portfolio_returns = self.daily_returns.dot(self.asset_weights)
        var = np.percentile(portfolio_returns.dropna(), (1 - confidence) * 100)

        return abs(var)


    def maximum_drawdown(self, absolute_value = True):
        """Method calculating the Maximum Drawdown (MDD)"""

        df_returns = self.get_portfolio_returns_df()
        df_cummax = df_returns.cummax()
        drawdowns = (df_returns - df_cummax) / df_cummax
        max_drawdown = drawdowns.min()
        if absolute_value:
            max_drawdown = abs(max_drawdown)
        return max_drawdown


    def portfolio_pe(self):
        """Method calculating the Value at Price to Earnings ratio (PE)"""

        pe = 0
        for i in self:
            pe += self[i]['object'].pe * self[i]['weight']
        return pe

    def portfolio_roe(self):
        """Method calculating the Value at Return on Equity (ROe)"""
        roe = 0
        for i in self:
            roe += self[i]['object'].roe * self[i]['weight']
        return roe
