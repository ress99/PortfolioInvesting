"""Module with algorithm implementations"""

import time
import random
from collections import Counter
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


    # Apply contraints and evaluate initial population
    obj.apply_constraints()
    obj.evaluate_population()

    # List to store the values of the individuals in the pareto fronts
    obj.pareto_fronts = [None] * obj.generations

    start_algo = time.perf_counter()
    for g in range(obj.generations):

        # A new generation
        start = time.perf_counter()

        # counter.update([tuple(sorted(i.asset_list)) for i in obj.pop])
        # print(len(counter), sum(counter.values()), counter.most_common(1)[0][1])

        # Select the offspring for the next generation
        offspring = obj.select(obj.pop, len(obj.pop))

        # Clone the selected individuals, AS or PO
        if not hasattr(obj, 'assets'):
            offspring = obj.clone_population(offspring)

        else:
            pop_weights = [i.asset_weights for i in offspring]
            offspring = list(obj.init_portfolio_individual
                             (assets = obj.assets, asset_weights = i) for i in pop_weights)

        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # Cross two individuals with probability CXPB
            if random.random() < obj.CXPB:
                obj.mate(child1, child2)

                # Delete fitness values of the children, to recalculate later
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation on the offspring
        for mutant in offspring:

            # Mutate an individual with probability MUTPB
            if random.random() < obj.MUTPB:
                obj.mutate(mutant)

                # Delete fitness values of the children, to recalculate later
                del mutant.fitness.values

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
        pvalues = np.array([i.fitness.values for i in pareto_front])
        pareto_front = [pareto_front[i] for i in np.argsort(pvalues[:, 0])]
        obj.pareto_fronts[g] = np.array([i.fitness.values for i in pareto_front])

        generation_info = f"Generation {g + 1} || Length of Pareto: {len(pareto_front)}"
        mean_info = f"Mean Return is {pvalues[:, 0].mean()} || Mean Variance is {pvalues[:, 1].mean()}"
        print(f"{generation_info} || {mean_info}")

        print(time.perf_counter() - start)


    print(
        len(pareto_front[0].cache),
        obj.pop_size,
        (time.perf_counter() - start_algo) / obj.generations
        )

    # Store the final pareto front
    obj.pareto_front = pareto_front
