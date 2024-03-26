
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from GSAToolkit import *
from initToolkit import *


"""
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", base=list, fitness=creator.FitnessMin,
                           speed=list, speed_limit=int, mass=float,
                              force=list, surrogated=int)
"""




def GSA_loop(n_generations, population_size, fitness_f,
              obj_type='minimize', normalize=False, gravity=parabolic_grav, train_size=1000,
              restart_tol=0.01,
              surrogated=None,
              random_seed=199,
              fitness_params=get_fitness_params()):

    toolbox = masterTool()
    animation_data = []

    if "shekel".__eq__(fitness_f.__name__):
        toolbox.register("evaluate", fitness_params["shekel"])
    elif "h1".__eq__(fitness_f.__name__):
        toolbox.register("evaluate", fitness_params["h1"])
    else:
        toolbox.register("evaluate", fitness_f)

    logbook, mstats = createLogbook()

    # print(["gen", "evals"] + mstats.fields)
    logbook = addChapters(logbook)

    random.seed(random_seed)
    # Create random population
    pplt = init_population(toolbox, population_size)
    # print(logbook)

    model_data = []
    enough = False

    for i in range(n_generations):
        # Dar masa a las particulas
        toolbox.updateMass(pplt, obj_type, normalize)
        # evaluate_fitness(pplt, fitness_f, obj_type, normalize)

        # Actualizar el valor de la gravedad
        toolbox.updateGrav(pplt, gravity, i)

        # Actualizar posicion y velocidad de la particula
        w = np.array([random.random() for i in range(population_size - 1)]).reshape(-1, 1)
        p = random.random()
        toolbox.updateXV(pplt, w, p)

        # Calcular fitness
        fitnesses = map(toolbox.evaluate, pplt)
        for part, fit in zip(pplt, fitnesses):
            part.fitness.values = fit

        # else:
        #    model = apply_surrogated(pplt, toolbox, surrogated, random_seed)
        #    for part, fit in zip(pplt, model.predict(np.array(pplt))):
        #        part.fitness.values = (fit,) # modelo devuelve valor
        if surrogated is not None:
            model_data += pplt
            # print(len(model_data))
            if (len(model_data) >= train_size) and (not enough):
                enough = True
                print(f"{fitness_f.__name__} cargando surrogado...")

                # entrenar el modelo
                model, substitutes = train_surrogated(model_data, surrogated, random_seed)

                for part, sub in zip(pplt, substitutes):
                    part.surrogated = sub
                    # print(part.surrogated )
                # toolbox.register("surrogated", model.predict)

            if enough:
                test = np.array([[p[0], p[1]] for p in pplt])
                substitutes = model.predict(test)
                for part, sub in zip(pplt, substitutes):
                    part.surrogated = sub
                    # print(part.surrogated )

        animation_data.append(np.array(pplt))


        record = mstats.compile(pplt)
        # print(record)

        #error = abs(record["fitness"]["std"] - previous_std)
        #previous_std = record["fitness"]["std"]

        logbook.record(gen=i + 1, evals=population_size, **record)
        # print(logbook)

    # print("record", record["std"])
    # if (restart_tol is not None) and (error < restart_tol):
    #   break
    # run+=1
    # pplt, logbook, previous_std = init_population(population_size, stats, logbook)

    return pplt, animation_data, logbook

"""
pplt, animation_data, log = main_loop(n_generations=100, population_size=30, fitness_f=plane, restart_tol=0.01,
                                      train_size=60,
                gravity=parabolic_grav, surrogated=(Pipeline([("MMS", MinMaxScaler()), ("SVR", SVR(max_iter=10000))]),
                                                    {'SVR__C': [1, 5, 10],
                                                     'SVR__kernel': ('linear', 'rbf')})
                                     )
"""
















