from deap import base
from deap import creator
from deap import tools
import random
from functools import partial
import numpy as np
import pandas as pd
from sp500 import SP500

sp500 = SP500()
from functools import partial
gen_idx = partial(random.sample, sp500.tickers_list, len(sp500.tickers_list))
tools.initIterate(list, gen_idx)

#Constants to use
number_cities = 50
number_generations = 250
#Crossover and mutation probabilities
CXPB, MUTPB = 0.5, 0.2
#Number of runs/seeds
iters = 30

class Zimbora:

    def __init__(self, asd):
        self.asd = asd

def dd():
    print('entrou')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#creator.create("Individual", list, fitness=creator.FitnessMin)
creator.create("Individual", Zimbora, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

#Creates an individual (a set with the number of cities)
toolbox.register("individual", tools.initIterate, creator.Individual, partial(random.sample, range(number_cities), number_cities))
#Creates the population by calling the individual function
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

pop = toolbox.population(n=40)
print(pop)