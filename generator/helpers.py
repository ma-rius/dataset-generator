import random
import time
import sys
import numpy as np
import pandas as pd
from deap import tools
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance


# ----  Dataset Creation ----
def create_dataset_and_or_mst(n=0, m=0, path='', covariance_between_attributes=False, m_groups=1, data=None):
    """
    Creates the Dataset and stores it
    :param n: the desired amount of instances
    :param m: the desired amount of features
    :param path: the path to store the dataset in
    :param covariance_between_attributes: True if the features shall be correlated
    :param m_groups: amount of groups
    :param data: if data already exists, the function uses this data to create a MST and omits the data set creation
    :return: the edges of the MST of the data
    """
    if data is None:
        data_ = pd.DataFrame()

        if covariance_between_attributes:
            if m % m_groups != 0:
                sys.exit('%i attributes can not be split into %i equal-sized groups' % (m, m_groups))
            m_per_group = int(m / m_groups)
            # initialize mean vectors
            all_means = []
            for g_mean in range(m_groups):
                mean = np.random.randint(100, size=m_per_group)
                all_means.append(mean)
            print('Mean Vector created.')

            # initialize covariance matrices (must be positive semi-definite)
            all_cov = []
            for g_cov in range(m_groups):
                A = np.random.rand(m_per_group, m_per_group)
                cov = np.dot(A, A.transpose())
                all_cov.append(cov)
            print('Covariance Matrix created.')

            # get values that follow the specified distribution
            for group in range(m_groups):
                data_ = pd.concat(
                    [data_, pd.DataFrame(np.random.multivariate_normal(all_means[group], all_cov[group], n))],
                    axis=1,
                    ignore_index=True)

        # for independent attributes:
        else:
            for attr in range(m):
                # concatenate columns: each column follows a normal distribution
                data_ = pd.concat(
                    [data_, pd.DataFrame(np.random.normal(random.randint(10, 100), np.random.rand(1) * 5, n))],
                    axis=1, ignore_index=True)
    else:
        data_ = data
        print('Now processing merged labels from sub data sets.')

    # add empty column for the labels
    data_['label'] = np.nan

    # save in csv file
    print('Store numbers in csv file...')
    data_.to_csv(path_or_buf=path, header=data_[:].columns.values.tolist(), index=False)

    # ----  End of Dataset Creation ----

    # build distance matrix
    start = time.time()
    print('Calculate Distance Matrix...')
    dist = np.triu(
        distance.cdist(data_[data_[:].columns.difference(['label'])], data_[data_[:].columns.difference(['label'])],
                       'euclidean'))
    print('Distance Matrix calculated in', time.time() - start, 'seconds.')

    # calculate Minimum Spanning Tree
    start = time.time()
    print('Calculate MST...')
    mst = minimum_spanning_tree(dist, overwrite=False).toarray()
    print('MST calculated in', time.time() - start, 'seconds.')

    # get row and column indices of non-zero values in mst
    # mst_edges has this form: [[0, 0], [1, 3], [1, 4], [3, 4]]
    print('Get row and column indices of non-zero values in MST...')
    # noinspection PyTypeChecker
    mst_edges = (np.argwhere(mst != 0)).tolist()
    return mst_edges


# ----  Complexity Measure ----
def complexity(individual, mst_edges, n_instances):
    # 1. Store the nodes of the spanning tree with different class.
    nodes = [-1] * n_instances
    for edge in mst_edges:
        if individual[edge[0]] != individual[edge[1]]:
            nodes[edge[0]] = 0
            nodes[edge[1]] = 0

    # 2. Compute the number of nodes of the spanning tree with different class.
    different = 0
    for i in range(n_instances):
        if nodes[i] == 0:
            different += 1
    return different / n_instances


def distance_to_desired_complexity(individual, mst_edges, n_instances, desired_complexity):
    """
    # calculates the difference of the actual and the desired complexity measure
    :param individual: one individual of the population
    :param mst_edges: the edges of an MST
    :param n_instances: the number of instances in the dataset
    :param desired_complexity: the desired complexity measure
    :return: difference of actual and desired complexity measure
    """
    # 1. Store the nodes of the spanning tree with different class.
    nodes = [-1] * n_instances
    for edge in mst_edges:
        if individual[edge[0]] != individual[edge[1]]:
            nodes[edge[0]] = 0
            nodes[edge[1]] = 0

    # 2. Compute the number of nodes of the spanning tree with different class.
    different = 0
    for i in range(n_instances):
        if nodes[i] == 0:
            different += 1
    complexity_ = abs(different / n_instances - desired_complexity)
    return complexity_


# ----  EA Fitness Function ----
def evaluate(individual, mst_edges, n_instances, desired_complexity):
    """
    The Fitness Function
    :param individual: the individual of the EA
    :param mst_edges: the edges of an MST
    :param n_instances: the number of instances in the dataset
    :param desired_complexity: the desired complexity measure
    :return: tuple of fitness for each optimization objective
    """
    fitness = ()  # fitness must be a tuple
    for mst_edges_ in mst_edges:
        # "append" each single fitness to fitness tuple
        fitness += (distance_to_desired_complexity(individual, mst_edges_, n_instances, desired_complexity),)
    # fitness += (abs(0.5 - (np.count_nonzero(individual)/n)),)  # "append" class share
    return fitness


# ########################################################
# the remainder of this code is copied from deap but
# implemented with a break condition to stop the EA
# when the fitness has reached a specified level
# ########################################################

def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    # for gen in range(1, ngen+1):
    gen = 1
    # while any(_ > 0.01 for _ in record['min']):
    while any(_ > 0.01 for _ in halloffame[0].fitness.values[0:3]):  # the break condition
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
        print('%.4f %.4f %.4f %.4f' % halloffame[0].fitness.values)
        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        gen += 1
    return population, logbook


# copied from deap
def eaSimple(population, toolbox, cxpb, mutpb, stats=None,
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

    # Begin the generational process
    gen = 1
    while record['min'] > 0.01:  # the break condition
    # while gen < 100:
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


# copied from deap
def varOr(population, toolbox, lambda_, cxpb, mutpb):
    assert (cxpb + mutpb) <= 1.0, ("The sum of the crossover and mutation "
                                   "probabilities must be smaller or equal to 1.0.")

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:  # Apply crossover
            ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:  # Apply reproduction
            offspring.append(random.choice(population))

    return offspring


# copied from deap
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


# copied from deap
def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(np.random.randint(size=size, low=imin, high=imax))
    ind.strategy = scls(np.random.randint(size=size, low=smin, high=smax))
    return ind


# copied from deap
def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children

        return wrappper

    return decorator
