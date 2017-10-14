import multiprocessing
import sys
import time

import numpy as np
import pandas as pd
from deap import creator, base, tools
from scipy import random
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance

total_time = time.time()


def main(b):
    # -------- Dataset Parameters --------
    m = 14  # number of attributes
    m_groups = 1  # number of independent groups the attributes are divided by
    n = 30000  # number of instances
    # b = 0.8  # the desired complexity, defined by the length of the class boundary, b ∈[0,1].
    # -------- End Dataset Parameters --------

    # check if attributes can be divided into specified amount of groups
    if m % m_groups != 0:
        sys.exit('%i attributes can not be split into %i equal-sized groups' % (m, m_groups))
    m_per_group = int(m / m_groups)

    # -------- GA Parameters --------
    population_size = 100  # int(n / 10)
    num_of_generations = 10000000
    # -------- End GA Parameters --------


    # set print options for large arrays
    np.set_printoptions(threshold=np.inf, precision=2, linewidth=np.inf)
    pd.set_option('expand_frame_repr', False)
    pd.set_option('compute.use_numexpr',
                  True)  # accelerating certain types of binary numerical and boolean operations using the numexpr library

    # ----  Dataset Creation ----
    start = time.time()

    # initialize mean vectors
    all_means = []
    for g_mean in range(m_groups):
        mean = np.random.randint(100, size=m_per_group)
        all_means.append(mean)
    # print('Mean Vectors: \n', all_means)
    print('Mean Vector created.')

    # initialize covariance matrices (must be positive semi-definite)
    all_cov = []
    for g_cov in range(m_groups):
        A = random.rand(m_per_group, m_per_group)
        cov = np.dot(A, A.transpose())
        all_cov.append(cov)
    # print(all_cov)
    print('Covariance Matrix created.')

    # get values that follow the specified distribution
    data = pd.DataFrame()
    for group in range(m_groups):
        data = pd.concat([data, pd.DataFrame(np.random.multivariate_normal(all_means[group], all_cov[group], n))],
                         axis=1,
                         ignore_index=True)

    # add empty column for the labels
    data['label'] = np.nan

    # save in csv file
    print('Store numbers in csv file...')
    data.to_csv(path_or_buf='../assets/data_%r.csv' %b, sep=';', header=data[:].columns.values.tolist(), index=False,
                decimal=',')

    # ----  End of Dataset Creation ----
    print('Data Creation done in:', time.time() - start, 'seconds.')

    # build distance matrix
    start = time.time()
    print('Calculate Distance Matrix...')
    dist = np.triu(
        distance.cdist(data[data[:].columns.difference(['label'])], data[data[:].columns.difference(['label'])],
                       'euclidean'))
    print('Distance Matrix calculated in', time.time() - start, 'seconds.')
    # print(dist)

    # calculate Minimum Spanning Tree
    start = time.time()
    print('Calculate MST...')
    mst = minimum_spanning_tree(dist, overwrite=True).toarray()
    print('MST calculated in', time.time() - start, 'seconds.')
    # print(mst)

    # get row and column indices of non-zero values in mst
    # mst_edges has this form: [[0, 0], [1, 3], [1, 4], [3, 4]]
    print('Get row and column indices of non-zero values in MST...')
    # noinspection PyTypeChecker
    mst_edges = (np.argwhere(mst != 0)).tolist()

    # print(mst_edges)


    # ---- GA ----



    # create Fitness Class
    # negative value for minimization problem (minimize the difference between specified and actual complexity)
    # -1 since we only have one objective to minimize with weight 100%
    toolbox = base.Toolbox()
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # create Individual class
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # toolbox = base.Toolbox()

    # one gene is one integer (0 or 1), the "class label"
    toolbox.register("gene", np.random.randint, 0, 2)  # randomly either 0 or 1

    # An individual is composed of n genes
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.gene, n=n)

    # A population is composed of many individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # GA techniques
    # Mutation
    toolbox.register("mutate", tools.mutFlipBit, indpb=1 / n)
    # toolbox.register("mutate", mutShuffleIndexes, indpb=1/n)
    # toolbox.register("mutate", mutUniformInt, low=0, up=1, indpb=1/n)
    # Crossover
    # A high eta will produce children resembling to their parents,
    # while a small eta will produce solutions much more different.
    toolbox.register("mate", tools.cxTwoPoint)

    # Selection
    toolbox.register("select", tools.selTournament, tournsize=int(n / 10))
    # toolbox.register("select", tools.selRandom)
    # register Fitness Function
    toolbox.register("evaluate", evaluate, n=n, b=b, mst_edges=mst_edges)

    # run the GA
    # noinspection PyTypeChecker

    print('GA started...')
    pop = toolbox.population(n=population_size)
    # print(pop)
    # ind1 = toolbox.individual()
    # print(ind1)
    # print(evaluate(ind1))
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    eaSimple(pop, toolbox=toolbox, cxpb=0.8, mutpb=1, ngen=num_of_generations, stats=stats, halloffame=hof,
             verbose=True)
    fitness_best_individual = evaluate(hof[0], mst_edges=mst_edges, n=n, b=b)
    # algorithms.eaMuCommaLambda(pop, toolbox, mu=10, lambda_=100,
    #     cxpb=0.6, mutpb=0.3, ngen=500, stats=stats, halloffame=hof)
    # algorithms.eaGenerateUpdate(toolbox, ngen=500, stats=stats, halloffame=hof, verbose=True)

    # print('Population:', pop)
    # print(log)
    print('------------------------------------')
    # print('Best individual:', hof[0])
    print('Fitness of best individual:', fitness_best_individual)
    print('Share of Class \'1\':', np.count_nonzero(hof[0]) / n)
    print('------------------------------------')
    data['label'] = hof[0]
    data.to_csv('../assets//data_%r.csv' %b, index=False)


# fitness function
def evaluate(individual, mst_edges, n, b):
    # number of edges connecting points of opposite classes is counted and divided by the
    # total number of connections. This ratio is taken as the measure of boundary length
    boundary_length = 0
    for edge in mst_edges:
        if individual[edge[0]] != individual[edge[1]]:
            boundary_length += 1
    fitness = abs(boundary_length / n - b)  # distance between actual and desired boundary length
    return fitness,


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    print(record)
    # Begin the generational process
    # for gen in range(1, ngen+1):
    gen = 1
    while record['min'] > 0:
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        gen += 1
        if verbose:
            print(logbook.stream)

    return population, logbook


def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


if __name__ == '__main__':
    start_ = time.time()
    toolbox = base.Toolbox()
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    complexities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for b in complexities:
        print('--------------------- COMPLEXITY:', b, '---------------------')
        main(b)
    print()
    print()
    print('Total time for all complexities:', time.time() - start_)

    # ---- End GA ----
