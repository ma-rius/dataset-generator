import random
import numpy as np
from deap import algorithms
from deap import creator, base, tools

n = 10
population_size = 10


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("gene", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.gene, n=n)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    return sum(individual),  # return as "1-tuple"

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1/n)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


def main():
    pop = toolbox.population(n=population_size)
    print(pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    # ind1 = toolbox.individual()
    # print(ind1)
    # print(evaluate(ind1))
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof, verbose=False)
    print()
    print(log)  # output format is prettier when printing log than with verbose=True
    # print('Population:', pop)
    # print(log)

if __name__ == '__main__':
    main()
