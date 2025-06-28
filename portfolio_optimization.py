"""Portfolio Optimization class with implementation of parent's methods and attributes"""

import os
import json
import logging

import config as c
from module import Module
import data_op as op

# Disable pylint warning for callint super() in __init__
# pylint: disable=super-init-not-called

# Set up logging
logger = logging.getLogger(__name__)


class PortfolioOptimization(Module):
    """Class with implementation of Module methods and attributes"""

    # Attributes for all Portfolio Optimization objects
    attributes_list = ['assets', 'pop', 'pareto_front', 'pareto_fronts', 'final_prtf']


    def __init__(self,
                 indexes = None,
                 prtf_size = None,
                 objectives = None,
                 start_date = None,
                 end_date = None,
                 CXPB = None,
                 MUTPB = None,
                 pop_size = None,
                 generations = None,
                 bb_path = None,
                 init_dict = None,
                 filename = None):
        """
        Initializes a PortfolioOptimization object.

        This constructor initializes the object using one of three methods:
        1. If a filename is provided, the attributes are loaded from a pickle file.
        2. If init_dict is provided, the attributes are extracted from it.
        3. If no filename or dictionary is provided, initialization is done from arguments.

        Args:
            Initialization attributes.

        Returns:
            None: The method initializes the object's attributes in place.
        """

        #Use Portfolio Optimization folder for object
        self.folder = c.po_folder

        #If a filename is provided, extract the attributes from the pickle file
        if filename is not None:
            self.init_from_file(filename)
            return

        #If an initialization dictionary is provided, extract the attributes from it
        if init_dict is not None:
            (indexes, prtf_size, objectives, start_date, end_date,
             bb_path, CXPB, MUTPB, pop_size, generations, _
             ) = self.get_init_data_from_dict(init_dict)

        #If no filename, initialize the attributes from the arguments
        self.init_attributes(indexes, prtf_size, objectives, start_date, end_date,
                                 bb_path, CXPB, MUTPB, pop_size, generations, filename)

        self.assets = None


    def init_population(self):
        """Initialize the population of individuals for the portfolio optimization algorithm."""
        if self.assets is None:
            logging.warning("Attempted to initialize population but 'assets' is not set.")
            raise ValueError("'assets' must be set before initializing the population.")

        self.pop = [self.init_portfolio_individual
                    (assets = self.assets,
                     asset_weights= None,
                     random_weights = True)
                    for _ in range(self.pop_size)]
        return

    def set_assets(self, assets = None, pkl_filename = None, json_filename = None):
        """
        Sets the assets for the portfolio optimization process.

        This method allows the user to set the assets in one of three ways:
        1. Directly by providing a list of assets.
        2. By loading assets from a pickle file.
        3. By loading assets from a JSON file.

        Args:
            assets (list, optional): A list of asset names to set. Defaults to None.
            pkl_filename (str, optional): The name of the pickle file (without the extension) 
                                        from which to load the assets. Defaults to None.
            json_filename (str, optional): The name of the JSON file (without the extension) 
                                        from which to load the assets. Defaults to None.

        Returns:
            None: The method modifies the `self.assets` attribute in place.
        """

        #If assets are provided as a list, set them
        if assets:
            self.assets = assets
        #If a pickle filename is provided, extract the assets from the pickle file
        elif pkl_filename:
            pickle_data = self.get_pickle_raw_data(pkl_filename, folder = c.as_folder)
            self.assets = self.get_assets_from_pickle(pickle_data)
        #If a JSON filename is provided, extract the assets from the JSON file
        elif json_filename:
            self.assets = self.set_assets_from_json(json_filename)

        else:
            print("Please set a filename or an asset list.")


    def get_assets_from_pickle(self, pickle_data):
        """
        Gets the assets from a pickle file and returns it as a list.

        Checks if the pickle data contains the final portfolio.
        If it does, extracts the asset list from the final portfolio.
        If not, returns an empty list.

        Args:
            pickle_data (dict): Dictionary containing data from the pickle file.

        Returns:
            list: List of assets extracted (if available) or an empty list.
        """

        #Check if the pickle data contains the final portfolio
        #Assets will be extracted from this portfolio
        if 'final_prtf' in pickle_data:
            final_prtf = pickle_data['final_prtf']
            return final_prtf.asset_list
        print('There is no final Portfolio in this pickle')
        return []


    def set_assets_from_json(self, json_filename):
        """
        Gets the assets from a JSON file and returns it as a list.

        Args:
            json_filename (str): The name of the JSON file

        Returns:
            list: List of assets extracted from the JSON file.
        """

        #Get the path to the JSON file
        folder_names = [c.prtf_folder, c.assets_folder, json_filename + '.json']
        #Open the JSON file and load its data
        with open(os.path.join(*folder_names), 'rb') as file:
            json_data = json.load(file)

        return json_data


    def clone(self, ind):
        """Clone an individual for the portfolio optimization algorithm."""

        #Get asset & weight list from individual
        assets = ind.asset_list
        asset_weights = ind.asset_weights
        #Initialize portfolio object with given assets and same weights
        return self.init_portfolio_individual(assets = assets, asset_weights = asset_weights)


    def get_data_from_pickle(self, pickle_data):
        """
        Creates new Asset Selection object with attributes from pickle file.

        Extracts initialization attributes, evolutionary operators,
        and additional attributes, and sets them on the new object.

        Args:
            pickle_data (dict): Dictionary containing data from the pickle file.

        Returns:
            None: The method modifies the object's attributes in place.
        """

        #Extracts initialization variables from a pickle dictionary
        (indexes, prtf_size, objectives, start_date, end_date,
         bb_path, CXPB, MUTPB, pop_size, generations, _
         ) = self.get_init_data_from_dict(pickle_data['init_data'])

        #Create new Asset Selection object with initialization attributes
        po = PortfolioOptimization(indexes, prtf_size, objectives, start_date, end_date,
                              CXPB, MUTPB, pop_size, generations, bb_path)
        
        # #Store evolutionary operators on Asset Selection object
        # po.ea_ops_from_dict(pickle_data)
        # #Store additional attributes on self object
        # po.set_attributes_from_pickle_dict(self.attributes_list, pickle_data)

        return po


    def create_test_portfolio_optimization(self, years = None, months = None, days = None):
        """
        Creates a new PO object for testing with a shifted time period.

        This method generates a new PO instance using the initialization data
        from the current object, but with the start and end dates shifted forward by the
        specified number of years, months, and days. The new object will have a population
        initialized with the same assets and weights as the original. If the original object
        has a Pareto front or a final portfolio, these are also copied to the new object.

        Args:
            years (int, optional): Number of years as a test period
            months (int, optional): Number of months as a test period
            days (int, optional): Number of days as a test period

        Returns:
            PortfolioOptimization: A new PO object with updated dates
            and copied population, pareto front, and final portfolio if present.
        """

        #Check if the original object has a final portfolio.
        #If not, does not make sense to create a new test object
        if not hasattr(self, 'final_prtf'):
            return None

        #If no time period is specified, default to 1 year
        if years is None and months is None and days is None:
            years = 1

        #Get init_data to update start and end dates
        init_data = self.init_data

        #Update start and end dates for the new PO object
        init_data['start_date'] = init_data['end_date']
        init_data['end_date'] = op.add_years_months_days(init_data['end_date'], years, months, days)

        #Create a new PO object with the updated initialization data
        new_po = PortfolioOptimization(init_dict = init_data)
        # Copy assets from the original object
        new_po.assets = self.assets

        #Initialize new population with individuals from the original population
        #Same assets and weights; new start and end dates
        new_po.pop = [new_po.init_portfolio_individual
                    (assets = ind.asset_list, asset_weights = ind.asset_weights)
                    for ind in self.pop]

        #If the original PO object has a pareto front, copy it to the new object
        #References the portfolio in the new Asset Selection object
        if hasattr(self, 'pareto_front'):
            #Get the indexes of the original pareto front individuals in the original population
            old_pareto_indexes = [self.pop.index(ind) for ind in self.pareto_front]
            #Get the pareto individuals according to theirindexes in the old population
            new_po.pareto_front = [new_po.pop[idx] for idx in old_pareto_indexes]

        #If the original AssetSelection object has a final portfolio, copy it to the new object
        #References the portfolio in the new Asset Selection object
        if hasattr(self, 'final_prtf'):
            #Get the index of the original final portfolio in the original population
            old_final_prtf_index = self.pop.index(self.final_prtf)
            #Get the final portfolio according to its index in the old population
            new_po.final_prtf = new_po.pop[old_final_prtf_index]

        #Return the new AssetSelection object
        return new_po
