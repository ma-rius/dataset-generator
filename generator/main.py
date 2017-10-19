import array
import multiprocessing
import numpy
import pickle
from deap import base
from deap import creator
from generator.functions import *

# -------- Dataset Parameters --------
n = 1000  # number of instances
m = 10  # number of attributes

# -------- GA Parameters --------
MIN_VALUE = 0  # individuals have int values [0.2), i.e. 0 or 1
MAX_VALUE = 2  # individuals have int values [0.2), i.e. 0 or 1
MIN_STRATEGY = 0.6  # min value for standard deviation of the mutation
MAX_STRATEGY = 1  # max value standard deviation of the mutation
population_size = 100  # number of individuals in each generation

# -------- Run Parameters --------
# complexity_measures = [0.9]
complexity_measures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
amount_of_datasets_per_complexity_measure = 1

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
    toolbox.register("evaluate", evaluate, mst_edges=mst_edges, n=n, b=b)

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
    eaSimple(pop, toolbox=toolbox, cxpb=0.85, mutpb=0.5, stats=stats,
             halloffame=hof,
             verbose=True)
    print('Best individual:', hof[0])
    print('Share of class 1:', np.count_nonzero(hof[0]) / n)
    # data = pd.read_csv(path, sep=';', decimal=',')

    # open file again
    # TODO it's ugly af
    data = pd.read_csv(path)

    # store labels
    data['label'] = [int(x) for x in hof[0]]
    data.to_csv(path, index=False)

    print('Time for this complexity:', time.time() - start_main)
    print('--------------------\n')


if __name__ == "__main__":
    start_total = time.time()
    # initialize multiprocessing
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # loop through amount of desired datasets for each complexity measure
    for i in range(amount_of_datasets_per_complexity_measure):
        start_iter = time.time()
        print('\n-------------------- ITERATION %r --------------------\n' % (i + 1))

        # loop through each complexity
        for complexity in complexity_measures:
            print('Complexity: %r\n' % complexity)

            # create data set (stores the file and returns the MST)
            data_set_mst = create_dataset(n=n, m=m, covariance_between_attributes=False,
                                          path='../assets/data_%r.csv' % complexity)
            # pickle.dump(data_set_mst, open('../assets/mst_edges.pkl', 'wb'))

            # data_set_mst = pickle.load(open('../assets/mst_edges.pkl', 'rb'))

            main(mst_edges=data_set_mst, b=complexity, path='../assets/data_%r.csv' % complexity)

        print('Time for iteration', (i + 1), ':', time.time() - start_iter)

    print('\n-------------------------------------------------')
    print('Total time:', time.time() - start_total)
