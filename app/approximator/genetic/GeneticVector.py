from app.approximator import Approximator
from app.approximator.Approximator import *




def fitness_func(window, x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, individual, table, smoothEdges, deviationType,
                 countEdges, realtime, savepath=None, func=None):
    # define your objective function here
    window.update()
    return \
    Approximator.smoothSeparate(x=x, y=y, y_real=y_real, yx=yx, yx2=yx2, x2=x2, x3=x3, x4=x4, func_lambd=func_lambd,
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


def select_parents(rankedSortedPopulation, num_parents):
    bestParents = [rankedSortedPopulation[i][1] for i in range(num_parents)]
    return bestParents


def crossover(parents, offspring_size, individual_size):
    offspring = []
    for i in range(offspring_size[0]):
        parent1_idx = i % len(parents)
        parent2_idx = (i + 1) % len(parents)
        newIndividual = [parents[parent1_idx][i] if random.uniform(0, 1) <= 0.5 else parents[parent2_idx][i] for i in
                         range(individual_size)]
        offspring.append(newIndividual)
    return offspring


def mutation(offspring, mutation_prob, shoulderLimit):
    individual_size = len(offspring[0])
    for i in range(len(offspring)):

        if np.random.rand() < mutation_prob:
            individual = [0]
            for j in range(1, individual_size - 1):
                dotsFromStart = j
                dotsToEnd = individual_size - j - 1
                individual.append(random.randint(1, min(dotsFromStart, dotsToEnd, shoulderLimit)))
            individual.append(0)
            offspring[i] = individual
        # for j in range(1, len(offspring[0]) - 1):
        #     if np.random.rand() < mutation_prob:
        #         shoulder_limit = min(shoulderLimit, min(j, len(offspring[i]) - j - 1))
        #
        #         # if np.random.rand() < 0.5:
        #         #     offspring[i][j] = offspring[i][j]+1 if offspring[i][j]+1 <= shoulder_limit else max(offspring[i][j] - 1,1)
        #         #
        #         #     #offspring[i][j] = np.random.randint(1, shoulder_limit) if shoulder_limit > 1 else 1
        #         # else:
        #         #     offspring[i][j] = offspring[i][j]-1 if offspring[i][j]-1 > 0 else min(offspring[i][j] + 1,shoulder_limit)
    return offspring


def genetic(progress, window, populationSize, num_generations, mutation_prob, individual_size, x, y, y_real, yx, yx2,
            x2, x3, x4, func_lambd, table, smoothEdges, deviationType, countEdges, realtime, savepath=None,
            shoulderLimit=math.inf):
    # parameters
    num_parents = populationSize // 2
    # initialize population
    population = generate_population(populationSize, individual_size, shoulderLimit)
    bests=[]
    deviations = []
    deviationsFromNoised = []
    deviationMin = float("inf")
    rankedPopulation = [(fitness_func(window, x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, individual, table=table,
                                   smoothEdges=smoothEdges, deviationType=deviationType, countEdges=countEdges,
                                   realtime=realtime, savepath=None),individual) for individual in population]
    rankedPopulation.sort()

    for i in range(num_generations):
        progress.config(value=i*100/num_generations)
        if Approximator.working:
            window.update()
            # evaluate fitness of population
            # select parents
            parents = select_parents(rankedPopulation, num_parents)

            # generate offspring
            offspring_size = (populationSize - num_parents, individual_size)
            offspring = crossover(parents, offspring_size,individual_size)

            # mutate offspring
            offspring = mutation(offspring, mutation_prob,shoulderLimit)
            # create new population
            offspring_fitness = [None] * len(offspring)
            rankedOffspring = [
                (fitness_func(window, x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, individual, table=table,
                             smoothEdges=smoothEdges, deviationType=deviationType, countEdges=countEdges,
                             realtime=realtime, savepath=None),individual) for individual in offspring]

            # combine parents and offspring

            rankedPopulation = rankedPopulation[:num_parents] + rankedOffspring
            rankedPopulation.sort()

            deviation, best_solution = rankedPopulation[0]#[0] Because it's sorted from least deviation to most

            print("deviation",deviation)
            #print("deviationFromNoised",deviationFromNoised)
            if deviationMin > deviation:
                deviationMin = deviation

            bests.append(deviationMin)
            deviations.append(deviation)
            #deviationsFromNoised.append(deviationFromNoised)

    # evaluate fitness of final population


    # get index of best solution

    # extract best solution
    best_solution = rankedPopulation[0][1]
    #plt.cla()
    #plt.plot(bests)
    #plt.show()
    # def fitf(solution, index):
    #     return fitness_func(x=x, y=y, y_real=y_real, yx=yx, yx2=yx2, x2=x2, x3=x3, x4=x4, func_lambd=func_lambd,
    #                                 individual=solution, table=table, smoothEdges=smoothEdges,
    #                                 deviationType=deviationType,
    #                                 countEdges=countEdges, realtime=realtime, savepath=None, window=window)
    # ga = pygad.GA(
    #     num_generations=num_generations,
    #     sol_per_pop=populationSize,
    #     num_genes=individual_size,
    #     gene_type=int,
    #     init_range_low=1,
    #     init_range_high=shoulderLimit,
    #     fitness_func=fitf,
    #     parent_selection_type="sss",
    #     crossover_type="single_point",
    #     mutation_type="random",
    #     mutation_percent_genes=20,
    #     keep_parents=1,
    #     num_parents_mating=10
    # )
    #
    # # Run the genetic algorithm
    # #ga.population = population
    # #print(population)
    # ga.run()
    #
    # # Get the best solution
    # best_solution = ga.best_solution()
    # best_fitness=0
    # print("Best solution found: ", best_solution)
    # print("Best fitness found: ", best_fitness)

    return best_solution, deviations, deviationsFromNoised
