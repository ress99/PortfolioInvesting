import random
import numpy as np
import time
from collections import Counter

def basic(object, asset_selection = True):


    object.evaluate_population()

    # Extracting all the fitnesses of

    object.pareto_fronts = [None] * object.generations
    test_counter = []

    for g in range(object.generations):
        start = time.perf_counter()
        # A new generation

        # Select the offspring for the next generation

        offspring = object.select(object.pop, len(object.pop))

        # Clone the selected individuals
        if not hasattr(object, 'assets'):
            pop_tickers = [i.asset_list for i in offspring]
            offspring = list(object.init_Portfolio_Individual(assets = i) for i in pop_tickers)
        else:
            pop_weights = [i.asset_weights for i in offspring]
            offspring = list(object.init_Portfolio_Individual(assets = object.assets, asset_weights = i) for i in pop_weights)


        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < object.CXPB:
                object.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < object.MUTPB:
                object.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = object.get_invalid_inds(offspring)
        object.evaluate_population(invalid_ind)

        # The population is entirely replaced by the offspring
        object.pop = object.select(object.pop + offspring, object.pop_size)

        pareto_front = object.find_non_dominant()
        
        # Gather all the fitnesses in one list
        pvalues = np.array([i.fitness.values for i in pareto_front])
        pareto_front = [pareto_front[i] for i in np.argsort(pvalues[:, 0])]
        object.pareto_fronts[g] = np.array([i.fitness.values for i in pareto_front])
        print("Generation %d || Length of Pareto: %d ||  Mean Return is %f || Mean Variance is %f" % (g + 1, len(pareto_front), pvalues[:, 0].mean(), pvalues[:, 1].mean()))
        end = time.perf_counter()
        print(end - start)

    object.pareto_front = pareto_front

