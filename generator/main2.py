import multiprocessing
import sys
import time

import numpy as np
import pandas as pd
from deap import algorithms
from deap import cma
from deap import creator, base, tools, algorithms
from scipy import random
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance
import array

# -------- Dataset Parameters --------
m = 14  # number of attributes
m_groups = 1  # number of independent groups the attributes are divided by
n = 1000  # number of instances
b = 0.1  # the desired complexity, defined by the length of the class boundary, b ∈[0,1].
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
    data = pd.concat([data, pd.DataFrame(np.random.multivariate_normal(all_means[group], all_cov[group], n))], axis=1,
                     ignore_index=True)

# add empty column for the labels
data['label'] = np.nan

# save in csv file
print('Store numbers in csv file...')
data.to_csv(path_or_buf='../assets/data.csv', sep=';', header=data[:].columns.values.tolist(), index=False, decimal=',')

# ----  End of Dataset Creation ----
print('Data Creation done in:', time.time() - start, 'seconds.')

# build distance matrix
start = time.time()
print('Calculate Distance Matrix...')
dist = np.triu(distance.cdist(data[data[:].columns.difference(['label'])], data[data[:].columns.difference(['label'])],
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

# fitness function
def evaluate(individual):
    # number of edges connecting points of opposite classes is counted and divided by the
    # total number of connections. This ratio is taken as the measure of boundary length
    boundary_length = 0
    for edge in mst_edges:
        if individual[edge[0]] != individual[edge[1]]:
            boundary_length += 1
    fitness = abs(boundary_length / n - b)  # distance between actual and desired boundary length
    return fitness,


# strategy = cma.Strategy(centroid=np.random.randint(size=n, low =0, high=2), sigma=5.0, lambda_=n)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", list)


def generateES(icls, scls):
    # ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind = icls(np.random.randint(size=n, low=0, high=2))
    ind.strategy = scls(np.random.randint(size=n, low=0, high=2))
    # ind.strategy = scls(np.random.rand(n))
    return ind


toolbox = base.Toolbox()
toolbox.register("individual", generateES, creator.Individual, creator.Strategy)


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


toolbox.register("evaluate", evaluate)

toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)

toolbox.decorate("mate", checkStrategy(0.5))
toolbox.decorate("mutate", checkStrategy(0.5))

# toolbox.register("gene", np.random.randint, 0, 2)  # randomly either 0 or 1
# toolbox.register("individual", tools.initRepeat, creator.Individual,
#                  toolbox.gene, n=n)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#
# toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mutate", tools.mutFlipBit, indpb=1 / n)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    print('GA started...')
    pop = toolbox.population(n=population_size)
    # ind1 = toolbox.individual()
    # print(ind1)
    # print(evaluate(ind1))
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=1, ngen=num_of_generations, stats=stats, halloffame=hof,
    #                     verbose=True)
    algorithms.eaMuCommaLambda(pop, toolbox, mu=n, lambda_=n,
                               cxpb=0.6, mutpb=0.3, ngen=500, stats=stats, halloffame=hof)
    # print('Population:', pop)
    # print(log)
    print('------------------------------------')
    print('Best individual:', hof[0])
    print('Fitness of best individual:', evaluate(hof[0]))
    print('------------------------------------')
    data['label'] = hof[0]
    data.to_csv('../assets/data.csv', index=False)


if __name__ == '__main__':
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    main()
