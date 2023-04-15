from deap import base
from deap import creator
from deap import tools
import random
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial
gen_idx = partial(random.sample, range(10), 10)
print(tools.initIterate(list, gen_idx))


#Constants to use
number_cities = 50
number_generations = 250
#Crossover and mutation probabilities
CXPB, MUTPB = 0.5, 0.2
#Number of runs/seeds
iters = 30



creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

#Creates an individual (a set with the number of cities)
toolbox.register("individual", tools.initIterate, creator.Individual, partial(random.sample, range(number_cities), number_cities))
#Creates the population by calling the individual function
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

###Calculates fitness of the individual
def eval_fit(individual):

    individual_array = np.asarray(individual) #Convert to array
    array_shifted = np.roll(individual_array, -1) #Create an array shifted by -1
    path = np.vstack((individual_array, array_shifted)) #Stack the arrays by column
    path = np.transpose(path)
    cost_array = city_cost_car[path[:,0], path[:,1]] #Create an array of the cost
    total_cost = np.sum(cost_array) #Sum it
    return total_cost, #Return fitness

###Performs an inversion mutation
def invert_mutation(individual, indpb):
    size = len(individual)

    if random.random() < indpb:
        bottom_index = random.randint(0, size - 2) #Choose bottom index
        top_index = random.randint(bottom_index + 1, size - 1) #Choose top index
        list = individual [bottom_index:top_index] #Clone from bottom to top index
        list.reverse() #Reverse List
        individual [bottom_index:top_index] = list #Update original individual
    return individual, #Return it


toolbox.register("evaluate", eval_fit) #Register evaluate function
toolbox.register("mate", tools.cxOrdered) #Register crossover function
toolbox.register("mutate", invert_mutation, indpb=0.6) #Register mutation function
toolbox.register("select", tools.selTournament, tournsize=3) #Register selection function

###Fuction that performs the evolution
def run_algorithm(iters):

    hall_of_fame_individuals = [None]*iters #To save the best individuals of each run
    hall_of_fame_fit = np.zeros(iters) #To save best fitnesses
    best_of_all = 100000 #Saves the best individual of all runs

    for iteration in range(iters): #Runs 30 times
        random.seed(iteration) #To allow reproducibility

        # create an initial population of 40 individuals (where
        # each individual is a list of integers)
        pop = toolbox.population(n=40)
        heu_path = toolbox.individual_heuristic()
        pop[0] = heu_path #Create heuristic individual and introduce to population


        #print("Start of evolution, with seed ", iteration)

        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        #print("  Evaluated %i individuals" % len(pop))

        # Extracting all the fitnesses of
        fits = [ind.fitness.values[0] for ind in pop]

        # Variable keeping track of the number of generations
        g = 0

        #To plot convergence curve, saves the mean of each generation
        Cost_Pop = np.zeros(number_generations+1);

        # Begin the evolution
        while min(fits) > 200 and g <= number_generations:
            # A new generation
            g = g + 1
            #print("-- Generation %i --" % g)

            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))

            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # cross two individuals with probability CXPB
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)

                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:

                # mutate an individual with probability MUTPB
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            #print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            #Keep the heuristic and the best individual from last generation, is it worth it?
            #pop[0] = heu_path
            #pop[1] = tools.selBest(pop, 1)[0]

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            #Gets some metrics of generation
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5

            '''print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)'''

            Cost_Pop[g-1] = mean       #Updates vector with mean of each generation (mean or min(fits))

        #print("-- End of (successful) evolution --")

        #Saves best individual and it's fitness
        best_ind = tools.selBest(pop, 1)[0]
        best_fit = best_ind.fitness.values[0]
        print("Best individual with seed %s is %s, %s\n" % (iteration, best_ind, best_ind.fitness.values))

        #If best of all, update
        if best_fit < best_of_all:
            best_cost_pop = Cost_Pop[0:g]
            best_of_all = best_fit

        hall_of_fame_individuals[iteration] = best_ind
        hall_of_fame_fit[iteration] = best_fit

    #Plot convergence curve
    plt.figure(1)
    plt.plot(best_cost_pop[0:g])
    plt.title('Convergence curve for the best run')
    plt.xlabel('Generations')
    plt.ylabel('Dist/Cost')
    plt.legend(['Mean'])
    plt.show()

    #Gets best of all time
    best_arg = np.argmin(hall_of_fame_fit)
    best_of_all = hall_of_fame_individuals[best_arg]

    #Prints best of all time, the stats for all the runs and the heuristic
    print("\n\nThe best of all is: ", best_of_all, best_of_all.fitness.values)
    print("\nThe mean of all the runs is ", np.mean(hall_of_fame_fit), "and the standard deviation ", np.std(hall_of_fame_fit))
    print ("\nThe heuristic is: ", heu_path, eval_fit(heu_path))

if __name__ == "__main__":
    #Calls algorithm
    run_algorithm(iters)
