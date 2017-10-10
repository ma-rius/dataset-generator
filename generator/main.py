import csv
import sys
# import random
from scipy import random
from scipy.spatial import distance
import pandas as pd
import numpy as np
import time
# import scipy.spatial.distance as dist
from joblib import Parallel, delayed
from progressbar import ProgressBar
from scipy.sparse.csgraph import minimum_spanning_tree

# -------- Dataset Parameters --------
m = 10  # number of attributes
m_groups = 2  # number of independent groups the attributes are divided by
n = 20  # number of instances
b = 50 / n  # the desired complexity, defined by the length of the class boundary, b ∈[0,1].
# l = [0, 1, 2]  # manually define a set of possible labels
# TODO make algorithm dependent on l or simply on max number of classes C
# TODO include parameter for minority class
# -------- End Dataset Parameters --------

# check if attributes can be divided into specified amount of groups
if m % m_groups != 0:
    sys.exit('%i attributes can not be split into %i equal-sized groups' % (m, m_groups))
m_per_group = int(m / m_groups)

# -------- GA Parameters --------
POPULATION_SIZE = 10
NUMB_OF_ELITE_CHROMOSOMES = 5  # those chromosomes will not be affected by crossover or mutation
TOURNAMENT_SELECTION_SIZE = 2
MUTATION_RATE = 0.05  # between 0 and 1, usually small
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

print(data)

# TODO change sep and decimal
data.to_csv(path_or_buf='../assets/data.csv', sep=';', header=data[:].columns.values.tolist(), index=False, decimal=',')

print('Data Creation and Saving Time:', time.time() - start)

# build distance matrix
start = time.time()
dist = np.triu(distance.cdist(data[data[:].columns.difference(['label'])], data[data[:].columns.difference(['label'])],
                              'euclidean'))
print('Distance Matrix Calculation Time:', time.time() - start)
print()
print(dist)

# calculate Minimum Spanning Tree
start = time.time()
mst = minimum_spanning_tree(dist, overwrite=True).toarray()
print('MST Calculation Time:', time.time() - start)

# class Chromosome:
#     def __init__(self):
#         self._genes = []
#         self._fitness = 0
#         i = 0
#         while i < n:
#             if random.random() >= 0.5:
#                 self._genes.append(1)
#             else:
#                 self._genes.append(0)
#             i += 1
#
#     def get_genes(self):
#         return self._genes
#
#     # TODO split into getter and recalc fitness
#     def get_fitness(self):
#         # number of edges connecting points of opposite classes is counted and divided by the
#         # total number of connections. This ratio is taken as the measure of boundary length
#         boundary_length = 0
#
#         for s in range(0, n, 1):
#             for t in range(s + 1, n, 1):  # take advantage of matrix symmetry
#                 if (mst[s][t] != 0) and (self._genes[s] != self._genes[t]):
#                     boundary_length += 1
#         self._fitness = 1 - abs((boundary_length / n) - b)  # distance between actual and desired boundary length
#         return self._fitness
#
#     # toString
#     def __str__(self):
#         return self._genes.__str__()
#
#
# class Population:
#     def __init__(self, size):
#         self._chromosomes = []
#         i = 0
#         while i < size:
#             self._chromosomes.append(Chromosome())
#             i += 1
#
#     def get_chromosomes(self):
#         return self._chromosomes
#
#
# class GeneticAlgorithm:
#     @staticmethod
#     def evolve(pop):
#         return GeneticAlgorithm._mutate_population(GeneticAlgorithm._crossover_population(pop))
#
#     @staticmethod
#     def _crossover_population(pop):
#         crossover_pop = Population(0)
#         for i in range(NUMB_OF_ELITE_CHROMOSOMES):
#             crossover_pop.get_chromosomes().append(pop.get_chromosomes()[i])
#         i = NUMB_OF_ELITE_CHROMOSOMES
#         while i < POPULATION_SIZE:
#             chromosome1 = GeneticAlgorithm._select_tournament_population(pop).get_chromosomes()[0]
#             chromosome2 = GeneticAlgorithm._select_tournament_population(pop).get_chromosomes()[0]
#             crossover_pop.get_chromosomes().append(GeneticAlgorithm._crossover_chromosomes(chromosome1, chromosome2))
#             i += 1
#         return crossover_pop
#
#     @staticmethod
#     def _mutate_population(pop):
#         for i in range(NUMB_OF_ELITE_CHROMOSOMES, POPULATION_SIZE):
#             GeneticAlgorithm._mutate_chromosome(pop.get_chromosomes()[i])
#         return pop
#
#     @staticmethod
#     def _crossover_chromosomes(chromosome1, chromosome2):
#         crossover_chrom = Chromosome()
#         for i in range(n):
#             if random.random() >= 0.5:
#                 crossover_chrom.get_genes()[i] = chromosome1.get_genes()[i]
#             else:
#                 crossover_chrom.get_genes()[i] = chromosome2.get_genes()[i]
#         return crossover_chrom
#
#     @staticmethod
#     def _mutate_chromosome(chromosome):
#         for i in range(n):
#             if random.random() < MUTATION_RATE:
#                 if random.random() < 0.5:
#                     chromosome.get_genes()[i] = 1
#                 else:
#                     chromosome.get_genes()[i] = 0
#
#     @staticmethod
#     def _select_tournament_population(pop):
#         tournament_pop = Population(0)
#         i = 0
#         while i < TOURNAMENT_SELECTION_SIZE:
#             tournament_pop.get_chromosomes().append(pop.get_chromosomes()[random.randrange(0, POPULATION_SIZE)])
#             i += 1
#         tournament_pop.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
#         return tournament_pop
#
#
# def _print_population(pop, gen_number):
#     print('\n----------------------')
#     print('Generation #', gen_number, '| Fittest chromosome fitness:', pop.get_chromosomes()[0].get_fitness())
#     print('----------------------')
#     # i = 0
#     # for x in pop.get_chromosomes():
#     #     print('Chromosome #', i, ' :', x, '| Fitness: ', x.get_fitness())
#     #     i += 1
#
#
# population = Population(POPULATION_SIZE)
# population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
# _print_population(population, 0)
# # start from first population
# generation_number = 1
# # while the fitness of the fittest chromosome in each population is smaller than the target chromosome's length
# while population.get_chromosomes()[0].get_fitness() < 1:
#     print('evolve population')
#     population = GeneticAlgorithm.evolve(population)
#     print('sort chromosomes')
#     population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
#     _print_population(population, generation_number)
#     generation_number += 1
#
#
# print('-------------')
# # print(population.get_chromosomes()[0])
#
# labels = np.array((population.get_chromosomes()[0]).get_genes())
# print(labels)
# print(len(labels))
# csv_input = pd.read_csv('../assets/data.csv')
# csv_input['label'] = labels
# csv_input.to_csv('../assets/data_with_labels.csv', index=False)
