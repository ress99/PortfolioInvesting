import data_op as op
import pandas as pd
import numpy as np
import talib as ta
from ticker import Ticker
from sp500 import SP500
from portfolio import Portfolio
import logging
import time
import config as c

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import random
from deap import base
from deap import creator
from deap import tools

sp500 = SP500()

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s: %(levelname)s: %(message)s'
                    )

PRTF_SIZE = 10
IND_SIZE = 10
START_DATE = '2022-10-01'
END_DATE = '2023-08-01'
CXPB, MUTPB = 0.5, 0.3
pop_size = 20
generations = 50
prtf_tickers = ['BG', 'FTNT', 'WRB', 'NTAP', 'FTRE', 'CSX', 'LVS', 'EMR', 'EL', 'WDC']

def initPortfolio(sp500 = sp500, nmbr_stocks = 10, tickers = None, weights = None):

    prtf = creator.Individual(sp500, 
                              cardinality_constraint=nmbr_stocks,
                              start_date = START_DATE, 
                              end_date = END_DATE)
    prtf.prtf_dict = tickers
    ticker_weights = [random.random() for _ in range(10)]
    prtf.ticker_weights = ticker_weights
    prtf.normalize_ticker_weights()
    if not op.check_valid_dates(prtf.prtf_df, START_DATE, END_DATE):
        prtf = initPortfolio()
    return prtf

def get_swap_numbers(lenght):

    nmbr1 = random.randint(0, lenght)
    nmbr2 = random.randint(0, lenght)

    if nmbr1 >= nmbr2:
        nmbr1, nmbr2 = get_swap_numbers(lenght)

    return nmbr1, nmbr2

def mate_prtfs(ind1, ind2):

    length = len(ind1.ticker_weights)
    nmbr1, nmbr2 = get_swap_numbers(length)

    weights1 = ind1.ticker_weights
    weights2 = ind2.ticker_weights

    swap1 = weights1[nmbr1:nmbr2]
    swap2 = weights2[nmbr1:nmbr2]
    weights1[nmbr1:nmbr2] = swap2
    weights2[nmbr1:nmbr2] = swap1

    ind1.ticker_weights = weights1
    ind2.ticker_weights = weights2

    return ind1, ind2

def mutate_prtf(ind):

    tic1 = random.sample(ind.tickers_list, 1)[0]
    tic2 = random.sample(ind.tickers_list, 1)[0]
    tic1_weight = ind.prtf_dict[tic1]['weight']
    tic2_weight = ind.prtf_dict[tic2]['weight']

    ind.change_ticker_weight(tic1, tic2_weight)
    ind.change_ticker_weight(tic2, tic1_weight)

    return ind

def evaluate_prtf(ind):

    prtf_return = ind.portfolio_return()
    prtf_variance = ind.portfolio_variance()

    ind.fitness.values = prtf_return, prtf_variance
    #ind.fitness.values = prtf_return,
    

creator.create("FitnessMax", base.Fitness, weights=(1.0,-1.0))
#creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", Portfolio, fitness=creator.FitnessMax)

t = base.Toolbox()
t.register("population", tools.initRepeat, list, initPortfolio)
t.register("mate", mate_prtfs)
t.register("mutate", mutate_prtf)
t.register("evaluate", evaluate_prtf)
t.register("select", tools.selNSGA2)
t.register("selBest", tools.selBest)

pop = [initPortfolio(tickers = prtf_tickers) for _ in range(pop_size)]
list(map(t.evaluate, pop))

# Extracting all the fitnesses of
fits = np.array([ind.fitness.values for ind in pop])
g = 0
Cost_Pop = np.zeros(generations+1)
a = 0
mean_returns = []
mean_var = []
pareto_front=tools.ParetoFront()

while g < generations:
    # A new generation
    g = g + 1

    # Select the offspring for the next generation
    offspring = t.select(pop, len(pop))

    # Clone the selected individuals
    pop_weights = [i.ticker_weights for i in offspring]
    offspring = list(initPortfolio(tickers = prtf_tickers, weights = i) for i in pop_weights)

    # Apply crossover and mutation on the offspring
    a = 0
    for child1, child2 in zip(offspring[::2], offspring[1::2]):

        # cross two individuals with probability CXPB
        if random.random() < CXPB:
            t.mate(child1, child2)

            # fitness values of the children
            # must be recalculated later
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:

        # mutate an individual with probability MUTPB
        if random.random() < MUTPB:
            t.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    list(map(t.evaluate, invalid_ind))

    # The population is entirely replaced by the offspring
    pop = t.select(pop + offspring, pop_size)
    #pop[:] = offspring

    # Gather all the fitnesses in one list and print the stats
    fits = np.array([ind.fitness.values for ind in pop])
    print("Generation %d || Size of Population: %d ||  Mean Return is %f || Mean Variance is %f" % (g, len(fits), fits[:, 0].mean(), fits[:, 1].mean()))

start = time.perf_counter()
pareto_front.update(pop)
pareto_front_values = np.array([i.fitness.values for i in pareto_front])
print("Generation %d || Length of Pareto: %d ||  Mean Return is %f || Mean Variance is %f" % (g, len(pareto_front), pareto_front_values[:, 0].mean(), pareto_front_values[:, 1].mean()))
end = time.perf_counter()
print(end - start)

fig = go.Figure()
fig.add_trace(go.Scatter(x=fits[:, 0], 
                        y=fits[:, 1], 
                        mode = 'markers',
                        marker = dict(size=8)
                        ))
fig.add_trace(go.Scatter(x=pareto_front_values[:, 0], 
                        y=pareto_front_values[:, 1], 
                        mode = 'markers + lines',
                        marker = dict(size=8)
                        ))
fig.update_layout(title='Portfolios',
                xaxis_title='Return',
                yaxis_title='Variance',
                width = 900,
                height = 700
                )
fig.show()

'''plt.figure()
plt.scatter(mean_returns, mean_var)
plt.title('Population on generation %d' % g)
plt.xlabel('Return')
plt.ylabel('Variance')
plt.show()'''
