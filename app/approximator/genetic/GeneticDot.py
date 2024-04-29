from math import log, inf

from app.approximator import Approximator
from app.approximator.Approximator import *


def fitness_func(index, x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, individual, table, smoothEdges, deviationType,
                 countEdges, realtime, savepath=None, func=None):
    shoulder_length = sum((2 ** ind) * bit for ind, bit in enumerate(individual))
    return Approximator.deviationForOneDot(shoulder_length, index, x, y, y_real, yx, yx2, x2, x3, x4, deviationType)


def generate_population(pop_size, chromosome_size, shoulderLimit):
    return [np.random.randint(2, size=chromosome_size) for i in range(pop_size)]


def select_parents(population, fitness_values, num_parents):
    fitness_values = np.array(fitness_values)
    return [population[i] for i in np.argpartition(fitness_values, num_parents)[:num_parents]]


def crossover(parents, offspring_size, chromosome_size):
    return [np.concatenate(
        (parents[i % len(parents)][:chromosome_size // 2], parents[(i + 1) % len(parents)][chromosome_size // 2:]),
        axis=0) for i in range(offspring_size[0])]


def mutation(offspring, mutation_prob, shoulderLimit):
    for individual in offspring:
        for j in range(individual.shape[0]):
            if np.random.uniform(0, 1) < mutation_prob:
                individual[j] = abs(individual[j] - 1)
                if sum([gen * 2 ** i for i, gen in enumerate(individual)]) > shoulderLimit:
                    individual = [0] * len(individual)
                    individual[0] = 1
    return offspring


def genetic(index, pop_size, num_generations, mutation_prob, x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, table,
            smoothEdges, deviationType, countEdges, realtime, savepath=None, shoulderLimit=inf):
    shoulderLimit = min(shoulderLimit, min(index, len(x) - index - 1))
    num_parents = pop_size // 2
    chromosome_size = int(log(shoulderLimit, 2))
    population = generate_population(pop_size, chromosome_size, shoulderLimit)
    for individual in population:
        if sum([gen * 2 ** i for i, gen in enumerate(individual)]) > shoulderLimit:
            individual = [0] * len(individual)
            individual[0] = 1
    bests = []
    errorMin = inf
    for i in range(num_generations):
        fitness_values = np.array([fitness_func(index, x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, individual, table,
                                                smoothEdges, deviationType, countEdges, realtime, savepath) for
                                   individual in population])
        parents = select_parents(population, fitness_values, num_parents)
        offspring_size = (pop_size - len(parents), chromosome_size)
        offspring = crossover(parents, offspring_size, chromosome_size)
        offspring = mutation(offspring, mutation_prob, shoulderLimit)
        population = np.concatenate((parents, offspring), axis=0)
        best_idx = np.argmin(fitness_values)

    # extract best solution
    best_solution = population[best_idx]
    # plt.cla()
    # plt.plot(bests)
    # plt.show()
    result = 0
    for ind, bit in enumerate(best_solution):
        result += (2 ** ind) * bit
    # print(result)
    return result
