import Approximator
from Approximator import *


def fitness_func(window,x, y, y_real, yx, yx2, x2, x3, x4, func_lambd, individual, table, smoothEdges, deviationType, countEdges, realtime, savepath=None, func=None):
    # define your objective function here
    window.update()
    return Approximator.smoothSeparate(x=x, y=y, y_real=y_real, yx=yx, yx2=yx2, x2=x2, x3=x3, x4=x4, func_lambd=func_lambd,
                                       shoulderVector=individual, table=table, smoothEdges=smoothEdges, deviationType=deviationType,
                                       countEdges=countEdges, realtime=realtime, savepath=None)[0]

def generate_population(population_size, individual_size, shoulder_limit):
    population = []
    for i in range(population_size):
        individual=[0]
        for j in range(1,individual_size-1):
            dotsFromStart = j
            dotsToEnd = individual_size - j - 1
            individual.append(random.randint(1,min(dotsFromStart,dotsToEnd,shoulder_limit)))
        individual.append(0)
        population.append(individual)

    return population

def genetic():
    population = generate_population()
    for i in range(iterations):

        rankedPopulation = [(fitness_func(individual), individual) for individual in population]
