"""Module with algorithm implementations"""

import time
import random
from collections import Counter

import logging
import numpy as np


counter = Counter()

def base_algorithm(obj):
    """
    Implementation of a base Genetic Algorithm.

    This function performs a genetic algorithm to optimize a portfolio. It iteratively evolves 
    a population of portfolio individuals over a specified number of generations, applying 
    selection, crossover, mutation, and constraints to improve the population's fitness.

    Args:
        obj: An object that encapsulates the genetic algorithm's parameters, methods, and 
             population. It must have the following attributes and methods:
             - Attributes:
                 - generations (int): Number of generations to run the algorithm.
                 - CXPB (float): Crossover probability.
                 - MUTPB (float): Mutation probability.
                 - pop (list): Current population of individuals.
                 - pop_size (int): Size of the population.
                 - assets (optional): List of assets for portfolio construction.

    Returns:
        None: The function modifies the `obj` in place by updating its population, Pareto fronts, 
              and final Pareto front.
    """

    # Check if the object has the required attributes
    aux_validate_object_attributes(obj)

    # Apply contraints and evaluate initial population
    obj.apply_constraints()
    obj.evaluate_population()

    # List to store the values of the individuals in the pareto fronts
    obj.pareto_fronts = [None] * obj.generations

    start_algo = time.perf_counter()
    for g in range(obj.generations):

        # A new generation
        start = time.perf_counter()

        counter.update([tuple(sorted(i.asset_list)) for i in obj.pop])
        logging.info(
            "Unique individuals: %d | Total: %d | Most common count: %d",
            len(counter), sum(counter.values()),
            counter.most_common(1)[0][1] if counter else 0)

        # Sort the population according to the fitness values
        offspring = obj.select(obj.pop, len(obj.pop))

        # Clone the selected individuals
        offspring = obj.clone_population(offspring)

        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # Crossover two individuals
            aux_apply_crossover(obj, child1, child2, random.random())

        # Apply mutation on the offspring
        for mutant in offspring:
            # Mutate an individual
            aux_apply_mutation(obj, mutant, random.random())

        # Normalize the weights of all the portfolios
        _ = [ind.normalize_asset_weights() for ind in offspring]

        # Apply constraints in the module
        obj.apply_constraints(offspring)
        # c = Counter([tuple(sorted(i.asset_list)) for i in offspring])

        # Evaluate the individuals without a valid fitness
        invalid_ind = obj.get_invalid_inds(offspring)
        obj.evaluate_population(invalid_ind)

        # Next generation is selected from both the actual population and offspring
        obj.pop = obj.select(obj.pop + offspring, obj.pop_size)

        # Find the pareto front of the population
        pareto_front = obj.find_non_dominant()

        # Gather all the fitnesses in one list and store the values
        pf_fitness = np.array([i.fitness.values for i in pareto_front])
        pareto_front = [pareto_front[i] for i in np.argsort(pf_fitness[:, 0])]
        obj.pareto_fronts[g] = pf_fitness

        generation_info = f"Generation {g + 1} || Length of Pareto: {len(pareto_front)}"
        mean_info = f"Mean Return: {pf_fitness[:, 0].mean()} || Mean Variance: {pf_fitness[:, 1].mean()}"
        print(f"{generation_info} || {mean_info}")

        print(f'Time to run generation:{time.perf_counter() - start}')


    print(pareto_front[0].get_cache_size(),
        (time.perf_counter() - start_algo) / obj.generations)

    # Store the final pareto front
    obj.pareto_front = pareto_front


def aux_validate_object_attributes(obj):
    """Validates that the object passed has the required attributes."""

    required_attrs = ['generations', 'CXPB', 'MUTPB', 'pop', 'pop_size']
    for attr in required_attrs:
        if not hasattr(obj, attr):
            raise AttributeError(f"Object passed to base_algorithm must have '{attr}'")


def aux_apply_crossover(obj, child1, child2, prob):
    """
    Apply crossover between two individuals with a given probability.
    If the crossover occurs, the fitness values of the children are deleted to be recalculated later.
    """

    # Cross two individuals with probability CXPB
    if prob < obj.CXPB:
        obj.mate(child1, child2)

        # Delete fitness values of the children, to recalculate later
        del child1.fitness.values
        del child2.fitness.values

def aux_apply_mutation(obj, mutant, prob):
    """Apply mutation on an individual with a given probability."""

    # Mutate an individual with probability MUTPB
    if prob < obj.MUTPB:
        obj.mutate(mutant)

        # Delete fitness values of the children, to recalculate later
        del mutant.fitness.values
