from PSOToolkit import *
from initToolkit import *

def PSO_loop(n_generations, population_size, fitness_f,
              obj_type='minimize', normalize=False, train_size=1000,
              restart_tol=0.01,
              surrogated=None,
              random_seed=199,
              fitness_params=get_fitness_params()):

    def shekel_arg0(part):
        return shekel(part, a=fitness_params["shekel"][0],
                      c=fitness_params["shekel"][1])

    toolbox = masterTool()
    animation_data = []

    if "shekel".__eq__(fitness_f.__name__):
        toolbox.register("evaluate", fitness_params["shekel"])
        fitness_f = fitness_params["shekel"]
    elif "h1".__eq__(fitness_f.__name__):
        toolbox.register("evaluate", fitness_params["h1"])
        fitness_f = fitness_params["h1"]
    else:
        toolbox.register("evaluate", fitness_f)

    logbook, mstats = createLogbook()

    # print(["gen", "evals"] + mstats.fields)
    logbook = addChapters(logbook)

    random.seed(random_seed)
    # Create random population
    pplt = init_population(toolbox, population_size)
    gbest = toolbox.initGBest(pplt)
    # print(logbook)

    model_data = []
    enough = False

    for i in range(n_generations):

        gbest = toolbox.update(pplt, fitness_f, gbest, kp=2, kg=2)

        # Calcular fitness
        fitnesses = map(toolbox.evaluate, pplt)
        for part, fit in zip(pplt, fitnesses):
            part.fitness.values = fit

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

    del toolbox

    return pplt, animation_data, logbook