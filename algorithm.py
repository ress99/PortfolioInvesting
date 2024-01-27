import random
import numpy as np
import time
from collections import Counter

def basic(obj):

    obj.apply_constraints()
    obj.evaluate_population()

    obj.pareto_fronts = [None] * obj.generations

    for g in range(obj.generations):
        start = time.perf_counter()
        
        # A new generation

        # Select the offspring for the next generation

        offspring = obj.select(obj.pop, len(obj.pop))

        # Clone the selected individuals
        if not hasattr(obj, 'assets'):
            pop_tickers = [i.asset_list for i in offspring]
            offspring = list(obj.init_Portfolio_Individual(assets = i) for i in pop_tickers)
        else:
            pop_weights = [i.asset_weights for i in offspring]
            offspring = list(obj.init_Portfolio_Individual(assets = obj.assets, asset_weights = i) for i in pop_weights)


        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < obj.CXPB:
                obj.mate(child1, child2)

                # fitness values of the children, to recalculate later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < obj.MUTPB:
                obj.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        [ind.normalize_asset_weights() for ind in offspring]

        obj.apply_constraints(offspring)
        # c = Counter([tuple(sorted(i.asset_list)) for i in offspring])
        
        invalid_ind = obj.get_invalid_inds(offspring)
        obj.evaluate_population(invalid_ind)

        # The population is entirely replaced by the offspring
        obj.pop = obj.select(obj.pop + offspring, obj.pop_size)

        pareto_front = obj.find_non_dominant()
        
        # Gather all the fitnesses in one list
        pvalues = np.array([i.fitness.values for i in pareto_front])
        pareto_front = [pareto_front[i] for i in np.argsort(pvalues[:, 0])]
        obj.pareto_fronts[g] = np.array([i.fitness.values for i in pareto_front])
        print("Generation %d || Length of Pareto: %d ||  Mean Return is %f || Mean Variance is %f" % (g + 1, len(pareto_front), pvalues[:, 0].mean(), pvalues[:, 1].mean()))
        print(time.perf_counter() - start)

    obj.pareto_front = pareto_front

