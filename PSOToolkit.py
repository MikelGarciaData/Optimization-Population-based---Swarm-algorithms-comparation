import random
from functools import reduce
from deap import base
from deap import tools
from initToolkit import *

def update_speed(population, fitness, gbest, kp=2, kg=2):
    w = random.random()
    for part in population:

        # Actualizar velocidad
        pbestdiff = [p - best for p, best in zip(part, part.pbest)]
        gbestdiff = [p - best for p, best in zip(part, gbest)]
        part.speed = [w * v + kp * pdiff + kg * gdiff for v, pdiff, gdiff in zip(part.speed, pbestdiff, gbestdiff)]

        # Comprobar que la velocidad no excede el limite
        part.speed = [value if value < part.speed_limit else part.speed_limit for value in part.speed]
        part.speed = [value if value > -part.speed_limit else -part.speed_limit for value in part.speed]
        # print(part.speed)

        # Actualizar posicion
        part[:] = [x - v for x, v in zip(part, part.speed)]

        # Actualizar pbest
        if fitness(part) < fitness(part.pbest):
            part.pbest = part

        # Actualizar fitness
        # part.fitness.values = (fitness(part),)

    # hallar mejor, actualizar gbest
    gbest = reduce(lambda x, y: x.pbest if fitness(x.pbest) < fitness(y.pbest) else y.pbest, population)

    return gbest

def getInitgbest(population):
        # return gbest
        return  reduce(lambda x, y: x.pbest if x.fitness.values[0] < y.fitness.values[0] else y.pbest, population)

def masterTool():
    toolbox = base.Toolbox()
    # Inicializacion de la poblacion
    toolbox.register("particle", init_particlePSO, size=2, part_min=-1, part_max=1, speed_limit=5)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("initGBest", getInitgbest)

    # Registrar metodos de actualizacion
    toolbox.register("update", update_speed)

    return toolbox