"""
Determine the labels in relation to the whole dataset (no sub datasets).
"""
import os
import array
import multiprocessing
import numpy
from deap import base
from deap import creator
from generator.helpers import *

# -------- Dataset Parameters --------
n = 1000  # number of instances
m = 6  # number of attributes

# -------- GA Parameters --------
MIN_VALUE = 0  # individuals have int values [0.2), i.e. 0 or 1
MAX_VALUE = 2  # individuals have int values [0.2), i.e. 0 or 1
MIN_STRATEGY = 0.2  # min value for standard deviation of the mutation
MAX_STRATEGY = 1  # max value standard deviation of the mutation
population_size = 100  # number of individuals in each generation

# -------- Run Parameters --------
complexity_measures = [0.2, 0.4, 0.6, 0.8]
amount_of_datasets_per_complexity_measure = 10

# set print options for large arrays
np.set_printoptions(threshold=np.inf, precision=2, linewidth=np.inf)
pd.set_option('expand_frame_repr', False)


# initialize EA
creator.create("FitnessMin", base.Fitness,
               weights=(-1.0,))  # -1 for "minimize" (the difference of desired and actual complexity)
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode='d')

toolbox = base.Toolbox()


# run the EA
def main(mst_edges, b, path):
    start_main = time.time()
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
                     n, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1/n)
    toolbox.register("select", tools.selTournament, tournsize=3)
    # toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate, mst_edges=mst_edges, n_instances=n, desired_complexity=b)

    toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
    toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))

    # generate initial population
    pop = toolbox.population(n=population_size)

    # store best individual of all evaluations in all populations and generations
    hof = tools.HallOfFame(1)

    # take care of some statistics of the population
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # run the EA
    eaSimple(pop, toolbox=toolbox, cxpb=0.85, mutpb=0.4, stats=stats,
             halloffame=hof,
             verbose=True)

    share_class_1 = np.count_nonzero(hof[0]) / n
    print('Share of class 1:', share_class_1)

    # open file again
    data = pd.read_csv(path)

    # store labels
    data['label'] = [int(x) for x in hof[0]]
    data.to_csv(path, index=False)

    print('Time for this complexity:', time.time() - start_main)
    print('--------------------\n')


if __name__ == '__main__':
    start_total = time.time()
    # initialize multiprocessing
    pool = multiprocessing.Pool()
    toolbox.register('map', pool.map)

    # loop through amount of desired datasets for each complexity measure
    for i in range(amount_of_datasets_per_complexity_measure):
        start_iter = time.time()
        print('\n-------------------- ITERATION %r --------------------\n' % (i + 1))

        # loop through each complexity
        for complexity in complexity_measures:
            print('Complexity: %r\n' % complexity)

            # create folder for each complexity measure if not existent
            if not os.path.exists('../assets/complexity_%r' % complexity):
                os.makedirs('../assets/complexity_%r' % complexity)

            # create data set (stores the file and returns the MST)
            data_set_mst = create_dataset_and_or_mst(n=n, m=m, covariance_between_attributes=True,
                                                     path='../assets/complexity_%r/data_%r.csv' % (complexity, (i + 1)))

            all_msts = [data_set_mst]  # here, we only have one MST (from the whole dataset), and no "sub" MSTs
            main(mst_edges=all_msts, b=complexity, path='../assets/complexity_%r/data_%r.csv' % (complexity, (i+1)))

        print('Time for iteration', (i + 1), ':', time.time() - start_iter)

    print('\n-------------------------------------------------')
    print('Total time:', time.time() - start_total)
