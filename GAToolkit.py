import random
from functools import reduce
from deap import base
from deap import tools
from initToolkit import *

def fMutation(offspring, mut_prob, toolbox):
    for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

def fCrossover(offspring, cx_prob, toolbox):
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values


def masterTool():
    # Configuramos los atributos de nuestra población y el individuo
    toolbox = base.Toolbox()
    toolbox.register("particle", init_particleGA, size=2, part_min=-10, part_max=10)
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 2)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)

    # Configuramos los operadores genéticos
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Operadores sobre la poblacion
    toolbox.register("mutations", fMutation)
    toolbox.register("crossover", fCrossover)

    return toolbox