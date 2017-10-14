import array
import multiprocessing
import numpy
from deap import base
from deap import creator

from generator.functions import *

# -------- Dataset Parameters --------
n = 100
m = 15

# -------- GA Parameters --------
MIN_VALUE = 0
MAX_VALUE = 2
MIN_STRATEGY = 0.5
MAX_STRATEGY = 1
population_size = 100

# -------- Run Parameters --------
# complexity_measures = [0.1]
complexity_measures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
amount_of_datasets_per_complexity_measure = 2

# initialize GA
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode='d')

toolbox = base.Toolbox()


def main(mst_edges, b, path):
    start_main = time.time()
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
                     n, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1 / n)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, mst_edges=mst_edges, n=n, b=b)

    toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
    toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))

    pop = toolbox.population(n=population_size)

    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    eaSimple(pop, toolbox=toolbox, cxpb=0.9, mutpb=0.4, stats=stats,
             halloffame=hof,
             verbose=True)
    # print('Best individual:', hof[0])
    data = pd.read_csv(path, sep=';', decimal=',')
    data['label'] = [int(x) for x in hof[0]]
    data.to_csv(path, index=False)

    print('Time for this complexity:', time.time() - start_main)
    print('--------------------\n')


if __name__ == "__main__":
    start_ = time.time()
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    for i in range(amount_of_datasets_per_complexity_measure):
        print('\n-------------------- ITERATION %r --------------------\n' % (i+1))
        for complexity in complexity_measures:
            print('Complexity: %r\n' % complexity)
            main(create_dataset(n=n, m=m, covariance_between_attributes=False,
                                path='../assets/data_%r.csv' % complexity), b=complexity,
                 path='../assets/data_%r.csv' % complexity)

    print('-------------------------------------------------')
    print('Total time:', time.time() - start_)
