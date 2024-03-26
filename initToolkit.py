import numpy as np

from deap.benchmarks import *
from deap import creator
from deap import tools

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# Funcion que devuelve un diccionario para hacer las cosas mas faciles
def get_fitness_params():

    def shekel_ac(part):
        return shekel(part, a=[[0.5, 0.5],
                          [0.25, 0.25],
                          [0.25, 0.75],
                          [0.75, 0.25],
                          [0.75, 0.75]],
                      c=[-0.002, -0.005, -0.005, -0.005, -0.005])

    def minh1(part):
        return (-h1(part)[0], )

    params = {"shekel": shekel_ac,
              "h1": minh1,
              "cx_prob": 0.6,
              "mut_prob": 0.3
              }
    return params


# FUNCIONES INICIALIZACION
def init_particleGSA(size=2, part_min=0, part_max=1, speed_limit=5):
    part = creator.Particle(random.uniform(part_min, part_max) for i in range(size))
    part.speed = [random.uniform(0, speed_limit) for _ in range(size)]
    part.speed_limit = speed_limit
    part.force = []
    part.surrogated = np.nan

    return part


def init_particlePSO(size=2, part_min=0, part_max=1, speed_limit=5):
    part = creator.Particle(random.uniform(part_min, part_max) for i in range(size))
    part.speed = [random.uniform(0, speed_limit) for _ in range(size)]
    part.speed_limit = speed_limit
    part.pbest = part
    # part.gbest = None
    part.surrogated = np.nan
    part.sur_pbest = np.nan

    return part


def init_particleGA(size=2, part_min=0, part_max=1, speed_limit=5):
    part = creator.Particle(random.uniform(part_min, part_max) for i in range(size))
    part.surrogated = np.nan

    return part


# particle = init_particle()


def init_population(toolbox, population_size):
    # Create random population
    pplt = toolbox.population(population_size)

    # Calcular fitness inicial
    fitnesses = map(toolbox.evaluate, pplt)
    for part, fit in zip(pplt, fitnesses):
        part.fitness.values = fit  # benchmarks devuelve (valor, )

    # Guardar estadisticas de inicializacion
    #record = stats.compile(pplt)
    #logbook.record(gen=0, evals=population_size, **record)

    # tomar anterior desviacion
    # print(record)
    #previous_std = record["fitness"]["std"]

    return pplt #, logbook, #previous_std


def createLogbook():
    stats = tools.Statistics(key=lambda part: part.fitness.values)
    srgts = tools.Statistics(key=lambda part: (part.surrogated,))
    resid = tools.Statistics(key=lambda part: (np.abs((part.fitness.values[0] - part.surrogated)/(10+part.fitness.values[0])),))

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    # stats.register("q1", lambda x: np.percentile(x, q=25))
    # stats.register("q2", lambda x: np.percentile(x, q=50))
    # stats.register("q3", lambda x: np.percentile(x, q=75))
    stats.register("min", np.min)
    stats.register("max", np.max)

    srgts.register("avg", np.mean)
    srgts.register("std", np.std)
    srgts.register("min", lambda x: np.nanmin(x) if np.any(~np.isnan(x)) else np.nan)
    srgts.register("max", lambda x: np.nanmax(x) if np.any(~np.isnan(x)) else np.nan)

    resid.register("avg", np.mean)
    resid.register("std", np.std)
    resid.register("min", lambda x: np.nanmin(x) if np.any(~np.isnan(x)) else np.nan)
    resid.register("max", lambda x: np.nanmax(x) if np.any(~np.isnan(x)) else np.nan)

    mstats = tools.MultiStatistics(fitness=stats, surrogated=srgts, error=resid)
    logbook = tools.Logbook()

    return logbook, mstats


def addChapters(logbook):
    logbook.header = "fitness", "surrogated", "error"
    logbook.chapters["fitness"].header = "gen", "evals", "avg", "std", "min", "max"
    logbook.chapters["surrogated"].header = "gen", "evals", "avg", "std", "min", "max"
    logbook.chapters["error"].header = "gen", "evals", "avg", "std", "min", "max"

    return logbook


# for part in pplt:
#    print(part.force)


# w = np.array([1 for i in range(POPULATION_SIZE-1)]).reshape(1,-1)
# random.seed(199)
# p = random.random()
# w = np.array([1 for i in range(POPULATION_SIZE - 1)]).reshape(1, -1)
# p = random.random()


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


# update_speed_position(pplt, weights=w, p=p)

# for part in pplt:
#    print(part)


# FITNESS Y SURROGADO
def calculate_fitness(pplt, toolbox, surrogated_data):
    if surrogated_data is None:
        fitnesses = map(toolbox.evaluate, pplt)
        for part, fit in zip(pplt, fitnesses):
            part.fitness.values = fit
    elif type(surrogated_data) == tuple:
        pass
        # entrenar con crossval
    else:
        pass
        # suponer que el usuario sabe lo que hace


def apply_surrogated(pplt, toolbox, surrogated_data, seed):
    surrogated, surrogated_params = surrogated_data

    X = np.array(pplt)

    X_train, X_test = train_test_split(X, train_size=0.5, random_state=seed,
                                       shuffle=True, stratify=None)

    scv = GridSearchCV(surrogated, param_grid=surrogated_params, cv=3)

    # print(X_train.shape, X_test.shape)

    fitnesses = map(toolbox.evaluate, X_train)
    y_train = np.array(list(fitnesses)).ravel()

    scv.fit(X_train, y_train)
    # print(scv.score(X_train, y_train))

    # print(scv.predict(X_test))

    return scv


def train_surrogated(model_data, surrogated_data, seed):
    surrogated, surrogated_params = surrogated_data

    data = np.array([[p[0], p[1], *p.fitness.values] for p in model_data])

    X_train, y_train = data[:, :-1], data[:, -1]

    scv = GridSearchCV(surrogated, param_grid=surrogated_params, cv=3)
    scv.fit(X_train, y_train)

    pred_out = scv.best_estimator_.predict(data[:, :-1])

    return scv.best_estimator_, pred_out
