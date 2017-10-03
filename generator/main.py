import csv
import random
import time

import numpy as np
import scipy.spatial.distance as dist
from joblib import Parallel, delayed
from progressbar import ProgressBar
from scipy.sparse.csgraph import minimum_spanning_tree

# -------- Dataset Parameters --------
m = 20  # number of features
n = 1000  # number of instances
b = 100 / n  # the desired complexity, defined by the length of the class boundary, b ∈[0,1].
l = [0, 1, 2]  # manually define a set of possible labels
# TODO make algorithm dependent on l
# TODO include parameter for minority class
# -------- End Dataset Parameters --------


# -------- GA Parameters --------
POPULATION_SIZE = 10
NUMB_OF_ELITE_CHROMOSOMES = 5  # those chromosomes will not be affected by crossover or mutation
TOURNAMENT_SELECTION_SIZE = 2
MUTATION_RATE = 0.05  # between 0 and 1, usually small
# -------- End GA Parameters --------


# set print options for large arrays
np.set_printoptions(threshold=np.inf, precision=2, linewidth=np.inf)

# -------- create and store dataset without labels --------
distribution_dict = {}
instances = {}

print('initialize distribution')
progress_bar_distribution = ProgressBar()
for j in progress_bar_distribution(range(0, m, 1)):
    # set distribution for feature j
    mu = random.randint(0, 100)  # TODO make specified number of attributes dependent on each other
    sigma = random.randint(0, 100)
    # print('Mu: %d, Sigma: %s' % (mu, sigma))
    distribution_dict.update({j: [mu, sigma]})

print('randomly pick values from distributions')
progress_bar_random = ProgressBar()
for i in progress_bar_random(range(0, n, 1)):
    instance = []
    for j in range(0, m, 1):
        instance.append(np.random.normal(distribution_dict[j][0], distribution_dict[j][1], 1)[0])
    instances.update({i: instance})

# store data in csv file
with open('../assets/data.csv', 'w') as f:
    print('write instances to file')
    time.sleep(0.1)
    progress_bar_file = ProgressBar()
    wtr = csv.writer(f, delimiter=',')  # TODO Semicolon as delimiter
    for index in progress_bar_file(instances):
        wtr.writerow(instances[index])
# ----------------------------------------------


# -------- build distance matrix --------
def calc_distances(p):
    row = []
    for q in range(0, p + 1, 1):
        row.append(0)
    for q in range(p + 1, n, 1):
        u, v = instances[p], instances[q]
        d = dist.euclidean(u, v)
        row.append(d)
    return row
print('calculate distances')
progress_bar_random = ProgressBar()
pool = Parallel(n_jobs=-1, verbose=1)
results = pool(delayed(calc_distances)(p) for p in progress_bar_random(range(0, n, 1)))

graph = np.array(results)
print(graph)
print()
# ----------------------------------------------

# -------- calculate Minimum Spanning Tree --------
print('calculate Minimum Spanning Tree')
#  noinspection PyTypeChecker
mst = minimum_spanning_tree(graph, overwrite=True).toarray()  # TODO check if overwrite = True has no bad influences
print(mst)


class Chromosome:
    def __init__(self):
        self._genes = []
        self._fitness = 0
        i = 0
        while i < n:
            if random.random() >= 0.5:
                self._genes.append(1)
            else:
                self._genes.append(0)
            i += 1

    def get_genes(self):
        return self._genes

    # TODO split into getter and recalc fitness
    def get_fitness(self):
        # number of edges connecting points of opposite classes is counted and divided by the
        # total number of connections. This ratio is taken as the measure of boundary length
        boundary_length = 0

        for s in range(0, n, 1):
            for t in range(s + 1, n, 1):  # take advantage of matrix symmetry
                if (mst[s][t] != 0) and (self._genes[s] != self._genes[t]):
                    boundary_length += 1
        self._fitness = 1 - abs((boundary_length / n) - b)  # distance between actual and desired boundary length
        return self._fitness

    # toString
    def __str__(self):
        return self._genes.__str__()


class Population:
    def __init__(self, size):
        self._chromosomes = []
        i = 0
        while i < size:
            self._chromosomes.append(Chromosome())
            i += 1

    def get_chromosomes(self):
        return self._chromosomes


class GeneticAlgorithm:
    @staticmethod
    def evolve(pop):
        return GeneticAlgorithm._mutate_population(GeneticAlgorithm._crossover_population(pop))

    @staticmethod
    def _crossover_population(pop):
        crossover_pop = Population(0)
        for i in range(NUMB_OF_ELITE_CHROMOSOMES):
            crossover_pop.get_chromosomes().append(pop.get_chromosomes()[i])
        i = NUMB_OF_ELITE_CHROMOSOMES
        while i < POPULATION_SIZE:
            chromosome1 = GeneticAlgorithm._select_tournament_population(pop).get_chromosomes()[0]
            chromosome2 = GeneticAlgorithm._select_tournament_population(pop).get_chromosomes()[0]
            crossover_pop.get_chromosomes().append(GeneticAlgorithm._crossover_chromosomes(chromosome1, chromosome2))
            i += 1
        return crossover_pop

    @staticmethod
    def _mutate_population(pop):
        for i in range(NUMB_OF_ELITE_CHROMOSOMES, POPULATION_SIZE):
            GeneticAlgorithm._mutate_chromosome(pop.get_chromosomes()[i])
        return pop

    @staticmethod
    def _crossover_chromosomes(chromosome1, chromosome2):
        crossover_chrom = Chromosome()
        for i in range(n):
            if random.random() >= 0.5:
                crossover_chrom.get_genes()[i] = chromosome1.get_genes()[i]
            else:
                crossover_chrom.get_genes()[i] = chromosome2.get_genes()[i]
        return crossover_chrom

    @staticmethod
    def _mutate_chromosome(chromosome):
        for i in range(n):
            if random.random() < MUTATION_RATE:
                if random.random() < 0.5:
                    chromosome.get_genes()[i] = 1
                else:
                    chromosome.get_genes()[i] = 0

    @staticmethod
    def _select_tournament_population(pop):
        tournament_pop = Population(0)
        i = 0
        while i < TOURNAMENT_SELECTION_SIZE:
            tournament_pop.get_chromosomes().append(pop.get_chromosomes()[random.randrange(0, POPULATION_SIZE)])
            i += 1
        tournament_pop.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
        return tournament_pop


def _print_population(pop, gen_number):
    print('\n----------------------')
    print('Generation #', gen_number, '| Fittest chromosome fitness:', pop.get_chromosomes()[0].get_fitness())
    print('----------------------')
    # i = 0
    # for x in pop.get_chromosomes():
    #     print('Chromosome #', i, ' :', x, '| Fitness: ', x.get_fitness())
    #     i += 1


population = Population(POPULATION_SIZE)
population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
_print_population(population, 0)
# start from first population
generation_number = 1
# while the fitness of the fittest chromosome in each population is smaller than the target chromosome's length
while population.get_chromosomes()[0].get_fitness() < 1:
    print('evolve population')
    population = GeneticAlgorithm.evolve(population)
    print('sort chromosomes')
    population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
    _print_population(population, generation_number)
    generation_number += 1


