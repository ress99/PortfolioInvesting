"""Module class with methods and attributes present in all modules"""

import random
import os
import importlib.util
import pickle
from collections import Counter

from functools import partial
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go
# import logging
# import time
# import datetime
# from deap import creator
# from deap import base

import data_op as op
import config as c
from portfolio import Portfolio
from constraints import Constraint
from index import Index


class Module:
    """Parent Class with methods and attributes present in all modules"""

    def __init__(self):
        """Placeholder initialization to be overridden by subclasses."""

        if type(self) is Module:
            raise TypeError("Module is an abstract class and cannot be instantiated directly.")

        self.pop = []
        self.folder = None
        self.attributes_list = []

        #Placeholder for pareto fronts and last pareto_front
        self.pareto_fronts = []
        self.pareto_front = []

        # Initialization attributes. Placeholders for the actual values.
        self.filename = None
        self.indexes = None
        self.prtf_size = None
        self.start_date = None
        self.end_date = None
        self.objectives = None
        self.CXPB = None
        self.MUTPB = None
        self.pop_size = None
        self.generations = None
        self.constraints = []
        self.bb_path = None
        self.bb = None  # For the blackbox module

        # Initialize evolutionary operator attributes
        self.mate = None
        self.mutate = None
        self.select = None
        self.evaluate = None
        self.algorithm = None


    def register(self, alias, function, *args, **kwargs):
        """
        Dynamically registers a new method to the instance with a specified alias.

        This method creates a new method for the instance by partially 
        applying the given function with the provided arguments and keyword 
        arguments. The new method is then bound to the instance under the specified alias.

        Args:
            alias (str): The name to assign to the dynamically created method.
            function (callable): The function to partially apply and bind as a method.
            *args: Positional arguments to pre-fill when the function is called.
            **kwargs: Keyword arguments to pre-fill when the function is called.

        Behavior:
            - The created method is accessible as an attribute of the instance using 
            the `alias` name.
            - The method retains the original function's docstring for documentation purposes.
            - The method's name is set to the `alias` for clarity and introspection.

        Example:
            def greet(name, greeting="Hello"):
                return f"{greeting}, {name}!"

            obj = Module()
            obj.register("say_hello", greet, greeting="Hi")
            print(obj.say_hello("Alice"))  # Output: "Hi, Alice!"
        """

        # Create a partially applied version of the function with the provided arguments
        partial_function = partial(function, *args, **kwargs)

        # Set the name and docstring of the partial function for clarity
        partial_function.__name__ = alias
        partial_function.__doc__ = function.__doc__

        # Dynamically add the new method to the instance with the specified alias
        setattr(self, alias, partial_function)


    def get_asset_weights(self, prtf, asset_weights, random_weights = False):
        """
        Calculate asset weights for Portfolio.
        If a list is provided as input, it will be used as the asset weights.
        If random_weights is True, random weights will be generated.
        If no input is provided, the weights will be equal for all assets.
        """

        #Apply specified weights
        if asset_weights is not None:
            prtf.asset_weights = asset_weights
            prtf.normalize_asset_weights()

        #Apply random weights if specified
        if random_weights:
            asset_weights = [random.random() for _ in range(len(prtf.asset_weights))]
            prtf.asset_weights = asset_weights
            prtf.normalize_asset_weights()

        #If neither case, apply the same weights to all assets
        else:
            prtf.apply_same_weights()

        return


    def init_portfolio_individual(
        self,
        assets = None,
        asset_weights = None,
        random_weights = False,
        start_date = None,
        end_date = None,
        indexes = None
        ):
        """
        Initializes an individual portfolio.

        Args:
            - assets (list, optional): List of asset names to include in the portfolio.
            Defaults to None. If None, a random selection of assets will be used.
            - asset_weights (list, optional): List of weights for the assets.
            Defaults to None. Given as input to get_asset_weights
            - random_weights (bool, optional): Whether to assign random weights to the assets.
            Defaults to False. Given as input to get_asset_weights
            - start_date (str, optional): Start date for the portfolio. Defaults to None.
            - end_date (str, optional): End date for the portfolio. Defaults to None.
            - indexes (list, optional): List of index objects to use. Defaults to None.

        Returns:
            Portfolio: A Portfolio object initialized with the specified parameters.
        """

        #If no input is provided, use the default values
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
        if indexes is None:
            indexes = self.index_objects

        #Create a new Portfolio object
        prtf = Portfolio(
            indexes,
            cardinality_constraint=self.prtf_size,
            start_date = start_date,
            end_date = end_date
            )

        prtf.fitness = self.objectives

        #If assets are not provided, randomly select them from the available assets
        if assets is None:
            prtf.prtf_dict = random.sample(self.all_assets, self.prtf_size)
        #If provided, use the given assets
        else:
            prtf.prtf_dict = assets

        #Assign weights to assets
        self.get_asset_weights(prtf, asset_weights, random_weights)

        return prtf


    def dominates(self, A, B):
        """
        Determines whether one solution dominates another in a multi-objective optimization context.

        A solution A dominates solution B if:
            - A is at least as good as B in all objectives.
            - A is strictly better than B in at least one objective.

        Args:
            A (iterable): A list or tuple representing the objective values of solution A.
            B (iterable): A list or tuple representing the objective values of solution B.

        Returns:
            bool: True if A dominates B, False otherwise.
        """

        strictly_better_boolean = False
        for a, b in zip(A, B):
            if a < b:
                # A is worse than B in at least one objective
                return False
            elif a > b:
                # A is better in at least one objective
                strictly_better_boolean = True

        return strictly_better_boolean


    def find_non_dominant(self):
        """
        Identifies the non-dominated individuals in a population 
        based on multi-objective optimization.

        A solution is considered non-dominated if no other solution in the population dominates it.

        Args:
            pop (list, optional): A list of individuals to evaluate. 
            If None, defaults to `self.pop`.

        Returns:
            list: A list of non-dominated individuals from the population.
        """

        non_dominant = []
        #Get population as input parameter

        for i, ind_a in enumerate(self.pop):
            is_dominated = False
            for j, ind_b in enumerate(self.pop):
                if i != j:
                    if self.dominates(ind_b.fitness.wvalues, ind_a.fitness.wvalues):
                        is_dominated = True
                        break
            if not is_dominated:
                non_dominant.append(ind_a)

        return non_dominant


    def get_invalid_inds(self, pop):
        """Gets invalid individuals from the population.
        Args:
            pop (list): Population of individuals."""
        return [ind for ind in pop if not ind.fitness.valid]


    def remove_invalids(self, pop = None):
        """Loops through population and removes invalid individuals."""

        return [ind for ind in pop if not ind.invalid]


    def get_pop(self, pop):
        """Retrieves population from object."""

        if pop is None:
            pop = self.pop
        return pop

    def evaluate_population(self, pop = None):
        """Gets population and evaluates it."""

        pop = self.get_pop(pop)
        list(map(self.evaluate, pop))

    #To be implemented in subclass
    def clone(self, ind):
        """Non-implemented method"""
        raise NotImplementedError("Subclasses must implement `clone`.")

    def clone_population(self, pop = None):
        """Clones population of object and returns it as a list"""

        pop = self.get_pop(pop)
        return [self.clone(ind) for ind in pop]

    def asset_list_counter(self, pop = None):
        """Sorts the asset list of the population and counts 
        the number of times each asset appears."""

        pop = self.get_pop(pop)
        counter = Counter([tuple(sorted(i.asset_list)) for i in pop])

        return counter


    def apply_constraints(self, pop = None):
        """For each individual in the population, applies the constraints
        and returns the population."""

        pop = self.get_pop(pop)
        for constraint in self.constraints:
            _ = [constraint.apply_constraint(ind) for ind in pop]
        return self.remove_invalids(pop)


    def get_object_from_pickle(self, filename, folder):
        """Reads pickle and extracts its data"""

        # Check if the folder is valid
        if folder not in [c.as_folder, c.po_folder]:
            print('Invalid folder')
            return

        #Gets raw pickle data
        pickle_data = self.get_pickle_raw_data(filename, folder)

        #Converts raw data into object from class and returns it
        return self.get_data_from_pickle(pickle_data)


    def get_pickle_raw_data(self, filename, folder = None):
        """Opens pickle file and returns the raw data"""

        # filename = filename + '.pkl'
        folder_names = [c.prtf_folder, folder, filename]
        with open(os.path.join(*folder_names), 'rb') as file:
            pickle_data = pickle.load(file)

        return pickle_data


    #To be implemented in subclass
    def get_data_from_pickle(self, pickle_data):
        """Non-implemented method"""
        raise NotImplementedError("Subclasses must implement `get_data_from_pickle`.")


    def save_to_pickle(self, filename = None):
        """
        Takes the name of the filename and the object's data.
        
        Saves the data to a pickle file in the specified folder.

        Args:
            filename (str): Name of the file to save the data to.
        
        Returns: 
            None
        """

        #Gets filename for pickle file
        pkl_filename = op.get_pickle_filename(filename)

        #Gets object data to save in pickle format
        data_to_pickle = self.get_data_to_pickle()

        #Saves the data to a pickle file
        pkl_filename = self.dump_pkl(data_to_pickle, pkl_filename)

        return pkl_filename


    def get_data_to_pickle(self):
        """
        Prepares the object's data for serialization into a pickle file.

        This method extracts the object's initialization data, evolutionary operators 
        and additional object attributes and saves them in a dictionary.

        Returns:
            dict: A dictionary containing the following keys:
                - "init_data": A dictionary with the object's initialization attributes.
                - "ea_ops": A dictionary with the evolutionary operators of the object.
                - Additional attributes specified in `self.attributes_list`.
        """

        #Extracts initial data and evolutionary operators from instance
        init_data = self.init_data_to_dict()
        ea_ops = self.ea_ops_to_dict()

        #Converts data to a dictionary
        data_to_pickle = {"init_data": init_data, 'ea_ops': ea_ops}

        #Adds additional object attributes to the dictionary
        data_to_pickle = self.add_obj_attributes_to_dict(self.attributes_list, data_to_pickle)

        return data_to_pickle


    # def dump_pkl(self, data_to_pickle, pkl_filename):
    #     """Saves data to a pickle file"""

    #     pkl_folder_names = [c.prtf_folder, self.folder, pkl_filename]
    #     with open(os.path.join(*pkl_folder_names), 'wb') as file:
    #         pickle.dump(data_to_pickle, file)

    #     return


    def dump_pkl(self, data_to_pickle, pkl_filename):
        """Saves data to a pickle file, incrementing the filename if it already exists."""

        base_folder = os.path.join(c.prtf_folder, self.folder)
        base_name, ext = os.path.splitext(pkl_filename)
        full_path = os.path.join(base_folder, pkl_filename)

        count = 0
        while os.path.exists(full_path):
            count += 1
            pkl_filename = f"{base_name}({count}){ext}"
            full_path = os.path.join(base_folder, pkl_filename)

        # Ensure the directory exists
        # os.makedirs(base_folder, exist_ok=True)

        with open(full_path, 'wb') as file:
            pickle.dump(data_to_pickle, file)

        return pkl_filename


    def init_data_to_dict(self):
        """
        Extracts initial data from the object and returns it as a dictionary.

        Returns:
            dict: Dictionary with the initialization attributes
        """

        init_data = {"indexes": list(self.indexes.values()), "prtf_size": self.prtf_size, "objectives": self.objectives, 
            "start_date": self.start_date, "end_date": self.end_date, 
            "bb_path": self.bb_path, "bb_mode": self.bb_mode, 
            "CXPB": self.CXPB, "MUTPB": self.MUTPB,
            "pop_size": self.pop_size, "generations": self.generations,
            "filename": self.filename}
        
        return init_data


    def ea_ops_to_dict(self):
        """
        Goes through evolutionary attributes of the object.

        If the attribute is present, it adds it to the dictionary.

        Returns:
            dict: Dictionary with the evolutionary attributes and their values.
        """
        ea_ops_data = {}

        if hasattr(self, 'mate'):
            ea_ops_data['mate'] = self.mate
        if hasattr(self, 'mutate'):
            ea_ops_data['mutate'] = self.mutate
        if hasattr(self, 'select'):
            ea_ops_data['select'] = self.select
        if hasattr(self, 'evaluate'):
            ea_ops_data['evaluate'] = self.evaluate
        if hasattr(self, 'algorithm'):
            ea_ops_data['algorithm'] = self.algorithm

        return ea_ops_data


    def add_obj_attributes_to_dict(self, attributes, data_to_pickle):
        """
        Adds attributes of the object to a dictionary.

        This method iterates through the object's attributes, check if they exist,
        and retrieves its value to the provided dictionary.

        Args:
            attributes (list): A list of attribute names (strings) to add to the dictionary.
            data_to_pickle (dict): The dictionary to which the attributes will be added.

        Returns:
            dict: The updated dictionary containing the specified attributes and their values.
        """

        #Loops through attributes
        for attr in attributes:
            #Checks their existance in the object
            if hasattr(self, attr):
                #If they exist, adds them to the dictionary
                data_to_pickle[attr] = getattr(self, attr)

        return data_to_pickle


    def init_from_file(self, filename):
        """
        Initializes self object from a pickle file.

        This method reads data from a pickle file, extracts initialization attributes, 
        evolutionary operators, and additional attributes, and sets them on the object.

        Args:
            filename (str): The name of the pickle file (without the `.pkl` extension) 
                            from which to load the object's data.

        Returns:
            None: The method modifies the object's attributes in place.
        """

        #Extracts raw data from pickle file
        pickle_data = self.get_pickle_raw_data(filename, self.folder)

        #Extracts initialization variables from pickle dictionary
        (indexes, prtf_size, objectives, start_date, end_date,
         bb_path, CXPB, MUTPB, pop_size, generations, _
         ) = self.get_init_data_from_dict(pickle_data['init_data'])

        #Store initialization attributes on self object
        self.init_attributes(indexes, prtf_size, objectives, start_date, end_date,
                             bb_path, CXPB, MUTPB, pop_size, generations, filename)

        #Store evolutionary operators on self object
        self.ea_ops_from_dict(pickle_data)

        #Store additional attributes on self object
        self.get_attributes_from_dict(self.attributes_list, pickle_data)

        return


    def get_init_data_from_dict(self, dictionary):
        """
        Extracts initialization variables from pickle dictionary.

        Args:
            dictionary (dict): Initialization dictionary

        Returns:
            tuple: A tuple containing all the initialization attributes.
        """

        #Extract certain attributes from the dictionary
        indexes = dictionary['indexes']
        prtf_size = dictionary['prtf_size']
        objectives = dictionary['objectives']
        start_date = dictionary['start_date']
        end_date = dictionary['end_date']

        #Check if additional attributes exist in the dictionary
        if 'pop_size' in dictionary.keys():
            pop_size = dictionary['pop_size']
        else:
            pop_size = None
        if 'generations' in dictionary.keys():
            generations = dictionary['generations']
        else:
            generations = None
        if 'filename' in dictionary.keys():
            filename = dictionary['filename']
        else:
            filename = None
        if 'bb_path' in dictionary.keys():
            bb_path = dictionary['bb_path']
        else:
            bb_path = None
        if 'CXPB' in dictionary.keys():
            CXPB = dictionary['CXPB']
        else:
            CXPB = None
        if 'MUTPB' in dictionary.keys():
            MUTPB = dictionary['MUTPB']
        else:
            MUTPB = None

        return (indexes, prtf_size, objectives, start_date, end_date,
                bb_path, CXPB, MUTPB, pop_size, generations, filename)


    def ea_ops_from_dict(self, pickle_data):
        """
        Loads evolutionary operators from pickle dictionary and assigns them to self object.

        This method checks if the provided dictionary contains evolutionary operators.. 
        If so, it assigns them to the corresponding attributes of the object.

        Args:
            pickle_data (dict): Dictionary containing data from a pickle file.

        Returns:
            None: The method modifies the object's attributes in place.
        """

        #If evolutionary operators are present in pickle, save dictionary
        if 'ea_ops' in pickle_data:
            ea_dict = pickle_data['ea_ops']
        else:
            return

        #For each operator, checks if they exist and store them in self object
        if 'mate' in ea_dict:
            self.mate = ea_dict['mate']
        if 'mutate' in ea_dict:
            self.mutate = ea_dict['mutate']
        if 'select' in ea_dict:
            self.select = ea_dict['select']
        if 'evaluate' in ea_dict:
            self.evaluate = ea_dict['evaluate']
        if 'algorithm' in ea_dict:
            self.algorithm = ea_dict['algorithm']

        return

    def init_attributes(self, indexes, prtf_size, objectives, start_date, end_date, 
                        bb_path, CXPB, MUTPB, pop_size, generations, filename):
        """
        Receives initialization attributes and stores them in self object

        Args:
            Initialization attributes.

        Returns:
            None: The method stores the object's attributes in place.
        """

        self.validate_attributes(indexes, prtf_size, objectives,
                                 start_date, end_date,
                                 CXPB, MUTPB, generations, bb_path)

        #Sets the attributes to the self object
        self.filename = filename
        self.indexes = indexes
        self.prtf_size = prtf_size
        self.start_date = start_date
        self.end_date = end_date
        self.objectives = objectives
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.pop_size = pop_size
        self.generations = generations
        self.constraints = []
        self.bb_path = bb_path
        if bb_path is not None:
            self.import_blackbox_module(bb_path)


    def validate_attributes(self, indexes, prtf_size, objectives, start_date, end_date, 
                            CXPB, MUTPB, generations, bb_path):
        """
        Validates the initialization attributes.

        Can be turned off using config file - validate_attributes

        Args:
            indexes (list): List of indexes.
            prtf_size (int): Portfolio size.
            objectives (tuple): Objectives for optimization.
            start_date (str): Start date.
            end_date (str): End date.
            CXPB (float): Crossover probability.
            MUTPB (float): Mutation probability.
            generations (int): Number of generations.

        Raises:
            TypeError: If any attribute is of the wrong type.
            ValueError: If any attribute has an invalid value.
        """

        if not c.validate_attributes:
            return

        # Check if indexes is a list
        if not isinstance(indexes, list):
            raise TypeError(f"'indexes' must be a list, got {type(indexes).__name__}.")
        # Check if indexes is a non-empty
        if not indexes:
            raise TypeError(f"'indexes' must be a non-empty list, got {indexes}.")
        # Check if all elements in the list are instances of the Index class
        if not all(isinstance(index, Index) for index in indexes):
            raise TypeError("All elements in 'indexes' must be instances of the 'Index' class.")

        # Check if prtf_size is an integer greater than 1
        if not isinstance(prtf_size, int) or prtf_size <= 1:
            raise ValueError(f"'prtf_size' must be an integer greater than 1, got {prtf_size}.")

        # Check if objectives is a tuple
        if not isinstance(objectives, tuple):
            raise TypeError(f"'objectives' must be a tuple, got {type(objectives).__name__}.")

        # Check if start_date is a valid date
        if not isinstance(start_date, (str)):
            raise TypeError(f"'start_date' must be a valid date string, got {type(start_date).__name__}.")

        # Check if end_date is a valid date and after start_date
        if not isinstance(end_date, (str)):
            raise TypeError(f"'end_date' must be a valid date string, got {type(end_date).__name__}.")
        if end_date <= start_date:
            raise ValueError(f"'end_date' must be after 'start_date'. Got start_date={start_date} and end_date={end_date}.")

        # Check if bb_path is a string or None
        if isinstance(bb_path, (str)):
            return
        if bb_path is not None:
            raise TypeError(f"'bb_path' must be a string or None, got {type(bb_path).__name__}.")

        # Check if CXPB is a float between 0 and 1
        if not isinstance(CXPB, float) or not 0 <= CXPB <= 1:
            raise ValueError(f"'CXPB' must be a float between 0 and 1, got {CXPB}.")

        # Check if MUTPB is a float between 0 and 1
        if not isinstance(MUTPB, float) or not 0 <= MUTPB <= 1:
            raise ValueError(f"'MUTPB' must be a float between 0 and 1, got {MUTPB}.")

        # Check if generations is an integer greater than 1
        if not isinstance(generations, int) or generations <= 1:
            raise ValueError(f"'generations' must be an integer greater than 1, got {generations}.")


    def get_attributes_from_dict(self, attributes, pickle_data):
        """
        Sets the object's attributes based on a dictionary.

        This method iterates through a list of attribute names, checks if each attribute exists 
        in the provided dictionary, and, if so, sets the corresponding attribute in the object.

        Args:
            attributes (list): A list of attribute names (strings) to check and set in the object.
            pickle_data (dict): A dictionary containing attribute names as keys and their values.
        
        Returns:
            None: The method modifies the object's attributes in place.
        """


        #Loops through attributes
        for attr in attributes:
            #Checks their existance in the object
            #If they exist, adds them to the instance
            if attr in pickle_data:
                setattr(self, attr, pickle_data[attr])


    def add_constraint(self, constraint):
        """
        Adds a constraint to the list of constraints.

        Args:
            constraint (Constraint): The constraint object to be added.

        Returns:
            None: The method modifies the `constraints` list in place.
        """

        #Checks if the constraint is a valid object
        if isinstance(constraint, Constraint):
            #If so, adds it to the list of constraints
            self.constraints.append(constraint)


    def normalize(self, value, min_val, max_val):
        """
        Scales value based on the provided minimum and maximum values, 
        returning a normalized value in the range [0, 1].
        """

        return (value - min_val) / (max_val - min_val)


    def run_method(self, method, *args, **kwargs):
        """
        Dynamically calls a method from the blackbox module (`bb`) with the provided arguments.

        Args:
            method (str): The name of the method to call from the blackbox module.
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            Any: The result of the called method.
        """

        #Checks if self has the blackbox module
        if hasattr(self.bb, method):
            # Get the method from self.bb
            method_to_call = getattr(self.bb, method)
            # Call the method with the provided arguments and keyword arguments
            return method_to_call(*args, **kwargs)
        else:
            raise AttributeError(f"{method} not found in bb.")


    def import_blackbox_module(self, module_path):
        """
        Imports a blackbox module from the specified file path.

        Args:
            module_path (str): The path to the blackbox module file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ImportError: If there is an error importing the module.
        """

        # Check if the file exists
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"The specified blackbox file does not exist: {module_path}")

        spec = importlib.util.spec_from_file_location("blackbox_module", module_path)
        self.bb = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(self.bb)
        except Exception as e:
            raise ImportError(f"An error occurred while importing the blackbox module: {e}") from e


    def set_bb_algorithm(self):

        if hasattr(self, 'bb'):
            self.algorithm = self.bb.algorithm
        else:
            print('Please select a BlackBox module.')

    @property
    def final_prtf(self):
        return self._final_prtf


    @final_prtf.setter
    def final_prtf(self, ind):

        if isinstance(ind, Portfolio):
            self._final_prtf = ind
        else:
            print('Please choose a valid individual')

    @final_prtf.deleter
    def final_prtf(self):
        if hasattr(self, '_final_prtf'):
            delattr(self, '_final_prtf')

    @property
    def indexes(self):
        return self._indexes


    @indexes.setter
    def indexes(self, idxs):
        self._indexes = {i.name: i for i in idxs}


    @property
    def index_objects(self):
        return list(self._indexes.values())

    @property
    def all_assets(self):
        assets = []
        aux_list = [list(i) for i in self.indexes.values()]
        _ = [assets.extend(i) for i in aux_list]
        return assets

    @property
    def bb_mode(self):
        if self.bb_path is None:
            return False
        else: 
            return True


    #Visualization methods

    def plot_min_max_product(self):

        composite = self.get_normalized_pareto_fronts()
        min_max = [[np.min(i), np.percentile(i, 25), np.mean(i), np.percentile(i, 75), np.max(i)] for i in composite]

        x_values = range(len(self.pareto_fronts))

        plt.plot(x_values, [i[0] for i in min_max], label='Min', color='blue')
        plt.plot(x_values, [i[2] for i in min_max], label='Mean', color='green')
        plt.plot(x_values, [i[4] for i in min_max], label='Max', color='red')

        plt.title('Minimum, Mean, and Maximum Return Values')
        plt.xlabel('Generations')
        plt.ylabel('Return')
        plt.grid()
        plt.legend()

        plt.show()

        return


    def get_normalized_pareto_fronts(self):

        all_returns = [individual[0] for generation in self.pareto_fronts for individual in generation]
        max_return, min_return = max(all_returns), min(all_returns)
        all_risks = [individual[1] for generation in self.pareto_fronts for individual in generation]
        max_risk, min_risk = max(all_risks), min(all_risks)

        composite = []
        for generation in self.pareto_fronts:
            normalized_returns = [self.normalize(individual[0], min_return, max_return) for individual in generation]
            normalized_risks = [self.normalize(individual[1], min_risk, max_risk) for individual in generation]

            composite_metric = [_return * risk for _return, risk in zip(normalized_returns, normalized_risks)]
            composite.append(composite_metric)
        
        return composite


    def standard_labels(self, title, xlabel, ylabel):

        if title is None:
            title = 'Objective Space'
        if xlabel is None:
            xlabel = 'Return'
        if ylabel is None:
            ylabel = 'Risk'

        return title, xlabel, ylabel


    def plot_paretos(self):

        fig = go.Figure()
        for i in self.pareto_fronts[:-1]:
            fig.add_trace(go.Scatter(x=i[:, 0],
                                    y=i[:, 1],
                                    mode = 'markers + lines',
                                    marker = dict(size=5, color='black')
                                    ))
        fig.add_trace(go.Scatter(x=self.pareto_fronts[-1][:, 0],
                                y=self.pareto_fronts[-1][:, 1],
                                mode = 'markers + lines',
                                marker = dict(size=8, color='red')
                                ))
        fig.update_layout(title='Portfolios',
                        xaxis_title='Return',
                        yaxis_title='Variance',
                        width = 900,
                        height = 700
                        )
        fig.show()


    def plot_min_max(self):

        min_max_return = [[i[0, 0], np.percentile(i[:, 0], 25), np.mean(i[:, 0]), np.percentile(i[:, 0], 75), i[-1, 0]] for i in self.pareto_fronts]
        min_max_risk = [[i[0, 1], np.percentile(i[:, 1], 25), np.mean(i[:, 1]), np.percentile(i[:, 1], 75), i[-1, 1]] for i in self.pareto_fronts]

        x_values = range(len(self.pareto_fronts))

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(x_values, [i[0] for i in min_max_return], label='Min', color='blue')
        plt.plot(x_values, [i[2] for i in min_max_return], label='Mean', color='green')
        plt.plot(x_values, [i[4] for i in min_max_return], label='Max', color='red')

        plt.title('Minimum, Mean, and Maximum Return Values')
        plt.xlabel('Generations')
        plt.ylabel('Return')
        plt.grid()
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(x_values, [i[0] for i in min_max_risk], label='Min', color='blue')
        plt.plot(x_values, [i[2] for i in min_max_risk], label='Mean', color='green')
        plt.plot(x_values, [i[4] for i in min_max_risk], label='Max', color='red')

        plt.title('Minimum, Mean, and Maximum Risk Values')
        plt.xlabel('Generations')
        plt.ylabel('Risk')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()

        return


    def remove_bounds(self, remove = False, min_values = None, max_values = None):
        """
        Remove out-of-bounds values from the pareto fronts.

        Args:
            remove (bool): Boolean indicating whether to remove out-of-bounds values.
            min_values (list, optional): Minimum values for each objective. Defaults to None.
            max_values (list, optional): Maximum values for each objective. Defaults to None.

        Returns:
            Any: Pareto fronts with out-of-bounds values removed if `remove` is True
        """

        #If min and max values were not provided, set them to default values
        if min_values is None:
            min_values = [0, 0]
        if max_values is None:
            max_values = [1000, 1000]

        #If remove is False, return the pareto fronts
        if not remove:
            return self.pareto_fronts

        #If remove is True, filter the pareto fronts based on the provided min and max values
        arr_list = [None] * len(self.pareto_fronts)
        for idx, arr in enumerate(self.pareto_fronts):
            arr_list[idx] = np.array([i for i in arr if
                                      all(min_v <= val <= max_v for val, (min_v, max_v) 
                                          in zip(i, zip(min_values, max_values)))])

        return arr_list


    def plot_2_objectives_as_sel_vs_po(self, as_sel, po, title = None, xlabel = None, ylabel = None):

        title, xlabel, ylabel = self.standard_labels(title, xlabel, ylabel)

        as_sel_arr_list, po_arr_list = as_sel.pareto_fronts, po.pareto_fronts

        plt.plot(as_sel_arr_list[-1][:, 0], as_sel_arr_list[-1][:, 1], color = 'blue', marker = 'o', label = 'Asset Selection Pareto Front')
        plt.plot(po_arr_list[-1][:, 0], po_arr_list[-1][:, 1], color = '#8dff52', marker = 'o', label = 'Portfolio Optimization Pareto Front')

        if hasattr(as_sel, 'final_prtf'):
            prtf_point = as_sel.final_prtf.fitness.values
            plt.plot(prtf_point[0], prtf_point[1], color = 'red', marker = 'o', label = 'Asset Selection Final Portfolio')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend()
        plt.show()


    def plot_2_objectives_as_sel_vs_2po(self, as_sel, po_without, po_with, title = None, xlabel = None, ylabel = None):

        title, xlabel, ylabel = self.standard_labels(title, xlabel, ylabel)

        as_sel_arr_list, po_withtout_arr_list, po_with_arr_list = as_sel.pareto_fronts, po_without.pareto_fronts, po_with.pareto_fronts

        plt.plot(as_sel_arr_list[-1][:, 0], as_sel_arr_list[-1][:, 1], color = 'blue', marker = 'o', label = 'Asset Selection Pareto Front')
        plt.plot(po_withtout_arr_list[-1][:, 0], po_withtout_arr_list[-1][:, 1], color = '#8dff52', marker = 'o', label = 'Portfolio Optimization Pareto Front No Constraint')
        plt.plot(po_with_arr_list[-1][:, 0], po_with_arr_list[-1][:, 1], color = '#FAFE18', marker = 'o', label = 'Portfolio Optimization Pareto Front 0.05 Constraint')

        if hasattr(as_sel, 'final_prtf'):
            prtf_point = as_sel.final_prtf.fitness.values
            plt.plot(prtf_point[0], prtf_point[1], color = 'red', marker = 'o', label = 'Asset Selection Final Portfolio')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend()
        plt.show()


    def plot_objective_space(self, arr_list = None, title = None, xlabel = None, ylabel = None):

        title, xlabel, ylabel = self.standard_labels(title, xlabel, ylabel)

        if arr_list is None:
            arr_list = self.pareto_fronts

        plt.figure()
        label = 'All Pareto Fronts'
        for idx, arr in enumerate(arr_list):
            if idx == len(arr_list) - 1:
                plt.plot(arr[:, 0], arr[:, 1], color = 'blue', marker = 'o', label = 'Final Pareto Front')
            else:       
                plt.plot(arr[:, 0], arr[:, 1], color = '#50ef1f', marker = 'o', label = label, alpha = 0.8)
                label = None

        if hasattr(self, 'final_prtf'):
            prtf_point = self.final_prtf.fitness.values
            plt.plot(prtf_point[0], prtf_point[1], color = 'red', marker = 'o', label = 'Final Portfolio')
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend()
        plt.show()


    def plot_objective_space_test(self, years = None, months = None, start_date = None, end_date = None, pareto_front = None, title = None, xlabel = None, ylabel = None):

        if pareto_front is None:
            #
            pareto_front = self.pareto_front

        if years is not None or months is not None:
            start_date = op.add_years_months_days(pareto_front[0].end_date, 0, 0, 1)
            end_date = op.add_years_months_days(start_date, years, months)

        elif start_date is None and end_date is None:
            return

        title, xlabel, ylabel = self.standard_labels(title, xlabel, ylabel)

        prtf_list = []
        for old_prtf in pareto_front:

            prtf = self.init_portfolio_individual(assets = old_prtf.asset_list, 
                                                  asset_weights = old_prtf.asset_weights,
                                                  start_date = start_date, end_date = end_date, 
                                                  indexes = old_prtf.index_objects)
            prtf_list.append(prtf)
            

        returns = [i.portfolio_return() for i in prtf_list]
        MDDs = [i.maximum_drawdown() for i in prtf_list]
        plt.scatter(MDDs, returns, color = 'blue', marker = 'o', label = 'Pareto Front')

        index_prtf = prtf_list[0].get_index_portfolio(prtf_list[0].index_objects, start_date, end_date)
        plt.scatter(index_prtf.maximum_drawdown(), index_prtf.portfolio_return(), color = 'red', marker = 'o', label = 'Index')

        plt.axvline(x=index_prtf.maximum_drawdown(), color='black', linestyle='--', lw=1.5)
        plt.axhline(y=index_prtf.portfolio_return(), color='black', linestyle='--', lw=1.5)

        plt.text(5, 5, 'Q1', fontsize=12, verticalalignment='center', horizontalalignment='center')
        plt.text(-5, 5, 'Q2', fontsize=12, verticalalignment='center', horizontalalignment='center')
        plt.text(-5, -5, 'Q3', fontsize=12, verticalalignment='center', horizontalalignment='center')
        plt.text(5, -5, 'Q4', fontsize=12, verticalalignment='center', horizontalalignment='center')


        plt.xlabel('Maximum Drawdown')
        plt.ylabel('Return')
        plt.title('Pareto Front Test Run')
        plt.legend()
        plt.grid(True)
        plt.show()

        self.calculate_quadrant_percentages([[MDDs[i], returns[i]] for i in range(len(MDDs))], [index_prtf.maximum_drawdown(), index_prtf.portfolio_return()])

    def calculate_quadrant_percentages(self, points, origin):
        
        quadrant_counts = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
        
        for point in points:
            relative_x = point[0] - origin[0]
            relative_y = point[1] - origin[1]
            
            if relative_x > 0 and relative_y > 0:
                quadrant_counts['Q1'] += 1
            elif relative_x < 0 and relative_y > 0:
                quadrant_counts['Q2'] += 1
            elif relative_x < 0 and relative_y < 0:
                quadrant_counts['Q3'] += 1
            elif relative_x > 0 and relative_y < 0:
                quadrant_counts['Q4'] += 1

        total_points = len(points)
        quadrant_percentages = {key: (value / total_points) * 100 for key, value in quadrant_counts.items()}
        print(quadrant_percentages)

        return quadrant_percentages


    # def plot_all_returns(self, pareto_front = None, start_date = None, end_date = None):

    #     if pareto_front is None:
    #         pareto_front = self.pareto_front
    #     if start_date is None:
    #         start_date = self.start_date
    #     if end_date is None:
    #         end_date = self.end_date

    #     #Do Clone
    #     first_label = True
    #     for old_prtf in pareto_front:
    #         prtf = Portfolio(list(old_prtf.indexes.values()), cardinality_constraint=old_prtf.cardinality_constraint, start_date = start_date, end_date = end_date)
    #         prtf.prtf_dict = old_prtf.asset_list
    #         prtf.asset_weights = old_prtf.asset_weights
    #         df_to_plot = prtf.get_portfolio_returns_df()
    #         if first_label:
    #             plt.plot(df_to_plot.index, df_to_plot.values, label = 'Portfolios', color = '#007be0', alpha = 0.8)
    #             first_label = False
    #         else:
    #             plt.plot(df_to_plot.index, df_to_plot.values, color = '#007be0', alpha = 0.65)

    #     index_prtf = old_prtf.get_index_portfolio(list(self.indexes.values()), start_date = start_date, end_date = end_date)
    #     df_to_plot = index_prtf.get_portfolio_returns_df()
    #     plt.plot(df_to_plot.index, df_to_plot.values, label = 'Index', color = 'red')


    #     plt.xlabel('Date')
    #     plt.ylabel('Return')
    #     # plt.title('Test Run Returns')
    #     plt.xticks(rotation=45)
    #     plt.subplots_adjust(bottom=0.2)
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()


    def plot_pareto_fronts_evolution(self, generations, arr_list = None):

        if arr_list is None:
            arr_list = self.pareto_fronts

        generations_array = [arr_list[i] for i in generations]
        colors = ['#add8e6', '#87ceeb', '#00bfff', '#1e90ff', '#0000cd']

        for idx, arr in enumerate(generations_array):
            plt.plot(arr[:, 0], arr[:, 1], color = colors[idx], marker = 'o', label = f"{generations[idx] + 1}th Generation")

        plt.title('Objective Space')
        plt.xlabel('Return')
        plt.ylabel('Risk')
        plt.grid()
        plt.legend()
        plt.show()
