import pygad
from app.approximator import Approximator
import math
import random

def genetic(progress, window, populationSize, num_generations, mutation_prob, individual_size, x, y, y_real, yx, yx2,
            x2, x3, x4, func_lambd, table, smoothEdges, deviationType, countEdges, realtime, savepath=None,
            shoulderLimit=math.inf):


    def fitness_func(individual, idx):
        # define your objective function here
        window.update()
        return \
            -1 * Approximator.smoothSeparate(x=x, y=y, y_real=y_real, yx=yx, yx2=yx2, x2=x2, x3=x3, x4=x4, func_lambd=func_lambd,
                                             shoulderVector=individual, table=table, smoothEdges=smoothEdges,
                                             deviationType=deviationType,
                                             countEdges=countEdges, realtime=realtime, savepath=None)[0]

    def generate_population(population_size, individual_size, shoulder_limit):
        population = []
        for i in range(population_size):
            individual = [0]
            for j in range(1, individual_size - 1):
                dotsFromStart = j
                dotsToEnd = individual_size - j - 1
                individual.append(random.randint(1, min(dotsFromStart, dotsToEnd, shoulder_limit)))
            individual.append(0)
            population.append(individual)
        return population

    initial_population = generate_population(populationSize, individual_size, shoulderLimit)

    fitness_function = fitness_func

    num_generations = num_generations
    num_parents_mating = 4

    num_genes = individual_size

    init_range_low = 1
    init_range_high = shoulderLimit

    parent_selection_type = "rws"
    keep_parents = 1

    crossover_type = "two_points"

    mutation_type = "random"
    mutation_percent_genes = mutation_prob*100

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           gene_type=int,
                           initial_population=initial_population)

    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    ga_instance.plot_fitness()
    return solution, None, None