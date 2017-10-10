import sys
import time

import numpy as np
import pandas as pd
from deap import algorithms
from deap import creator, base, tools
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance
from scipy import random

# -------- Dataset Parameters --------
m = 10  # number of attributes
m_groups = 2  # number of independent groups the attributes are divided by
n = 5  # number of instances
b = 0.9  # the desired complexity, defined by the length of the class boundary, b ∈[0,1].
# -------- End Dataset Parameters --------

# check if attributes can be divided into specified amount of groups
if m % m_groups != 0:
    sys.exit('%i attributes can not be split into %i equal-sized groups' % (m, m_groups))
m_per_group = int(m / m_groups)

# -------- GA Parameters --------
population_size = 1  # int(n / 10)
num_of_generations = 100
# -------- End GA Parameters --------


# set print options for large arrays
np.set_printoptions(threshold=np.inf, precision=2, linewidth=np.inf)
pd.set_option('expand_frame_repr', False)

start = time.time()
# initialize mean vectors
all_means = []
for g_mean in range(m_groups):
    mean = np.random.randint(100, size=m_per_group)
    all_means.append(mean)

# print('Mean Vectors: \n', all_means)

# initialize covariance matrices (must be positive semi-definite)
all_cov = []
for g_cov in range(m_groups):
    A = random.rand(m_per_group, m_per_group)
    cov = np.dot(A, A.transpose())
    all_cov.append(cov)

# print(all_cov)

# create data
data = pd.DataFrame()
for group in range(m_groups):
    data = pd.concat([data, pd.DataFrame(np.random.multivariate_normal(all_means[group], all_cov[group], n))], axis=1,
                     ignore_index=True)

data['label'] = np.nan

# print(data)

data.to_csv(path_or_buf='../assets/data.csv', sep=';', header=data[:].columns.values.tolist(), index=False, decimal=',')

# print('Data Creation and Saving Time:', time.time() - start)

# build distance matrix
# start = time.time()
dist = np.triu(distance.cdist(data[data[:].columns.difference(['label'])], data[data[:].columns.difference(['label'])],
                              'euclidean'))
# print('Distance Matrix Calculation Time:', time.time() - start)
# print()
# print(dist)

# calculate Minimum Spanning Tree
# start = time.time()
mst = minimum_spanning_tree(dist, overwrite=True).toarray()
# print('MST Calculation Time:', time.time() - start)
print(mst)
# print(data['label'])

# ---- GA ----
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("gene", np.random.randint, 0, 2)  # randomly either 0 or 1
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.gene, n=n)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# fitness function
def evaluate(individual):
    # number of edges connecting points of opposite classes is counted and divided by the
    # total number of connections. This ratio is taken as the measure of boundary length
    boundary_length = 0

    for s in range(0, n, 1):
        for t in range(s + 1, n, 1):  # take advantage of matrix symmetry
            if (mst[s][t] != 0) and (individual[s] != individual[t]):
                boundary_length += 1
    fitness = abs(boundary_length / n - b)  # distance between actual and desired boundary length
    return fitness,


toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1 / n)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


def main():
    pop = toolbox.population(n=population_size)
    # print(pop)
    # ind1 = toolbox.individual()
    # print(ind1)
    # print(evaluate(ind1))
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=num_of_generations, stats=stats, verbose=1)

    print('Population:', pop)
    # print(log)

    # data['label'] = pop
    # data.to_csv('../assets/data_with_labels.csv', index=False)


if __name__ == '__main__':
    main()
    # ---- End GA ----
