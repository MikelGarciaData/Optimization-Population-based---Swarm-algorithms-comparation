import numpy as np
from deap import base
from deap import tools
from initToolkit import *



def R_ij(part1, part2):
    # Calculate distance between particles
    return sum([np.sqrt(part1[0]**2 + part2[0]**2),
            np.sqrt(part1[1]**2 + part2[1]**2)])



def update_mass(population, obj_type='minimize', normalize=False):
    fitness_list = []

    # Evaluar el fitness de cada particula
    for part in population:
        # fitness = fitness_f(part)
        part.mass = part.fitness.values[0]
        fitness_list.append(part.mass)

    # hallar el mejor y el peor
    if 'minimize'.__eq__(obj_type):
        best = min(fitness_list)
        worst = max(fitness_list)
    else:
        best = max(fitness_list)
        worst = min(fitness_list)

    # Relativizar las masas
    fitness_list = []
    for part in population:
        part.mass = (part.mass - worst) / (best - worst)

    # En caso de que queramos normalizar
    if normalize:
        for part in population:
            part.mass = part.mass / sum(fitness_list)


def constant_grav(t):
    return 9.8


def parabolic_grav(t):
    return 100 / (t + 1) ** 2


# definimos una funcion que toma la poblacion y un parametro gravedad
def gravity_update(population, gravity, t):
    for part1 in population:
        for part2 in population:
            if part1 != part2:
                # Calcular distancia entre particulas
                dist = R_ij(part1, part2)
                # Calcular direccion de la fuerza
                direction = [(y - x) for x, y in zip(part1, part2)]

                # Calcular fuerza/masa (acceleracion) de la particula j sobre particula i
                a_ij = [(gravity(t) * part2.mass) / dist * x for x in direction]
                # Añadimos la fuerza trampeada
                part1.force.append(a_ij)


# Combinacion lineal de las fuerzas trampeadas
def update_speed_position(population, weights, p, random_seed=199):
    for part in population:
        # Calculamos la aceleracion
        actn = np.sum(np.array(part.force) * weights, axis=0)

        # Actualizamos velocidad
        part.speed = [p * v + a for v, a in zip(part.speed, actn)]

        # Actualizamos posicion
        part[:] = [x + v for x, v in zip(part, part.speed)]

        # restablecer fuerza para que no se acumule
        part.force = []

# evaluate_fitness(pplt, rosenbrock, obj_type='minimize', normalize=False)
#gravity_update(pplt, constant_grav, t=1)

def masterTool():
    toolbox = base.Toolbox()
    # Inicializacion de la poblacion
    toolbox.register("particle", init_particleGSA, size=2, part_min=-1, part_max=1, speed_limit=5)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)

    # Registrar metodos de actualizacion
    toolbox.register("updateGrav", gravity_update)
    toolbox.register("updateMass", update_mass)
    toolbox.register("updateXV", update_speed_position)

    return toolbox