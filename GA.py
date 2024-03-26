from GAToolkit import *
from initToolkit import *

def GA_loop(n_generations, population_size, fitness_f,
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

    cx_prob = fitness_params["cx_prob"]
    mut_prob = fitness_params["mut_prob"]

    model_data = []
    enough = False

    if "shekel".__eq__(fitness_f.__name__):
        toolbox.register("evaluate", fitness_params["shekel"])
    elif "h1".__eq__(fitness_f.__name__):
        toolbox.register("evaluate", fitness_params["h1"])
    else:
        toolbox.register("evaluate", fitness_f)

    logbook, mstats = createLogbook()

    # print(["gen", "evals"] + mstats.fields)
    logbook = addChapters(logbook)

    # Generamos nuestra población inicial y ejecutamos el algoritmo genético
    population = init_population(toolbox, population_size)

    for i in range(n_generations):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        toolbox.mutations(offspring, mut_prob, toolbox)
        toolbox.crossover(offspring, cx_prob, toolbox)

        # Evaluamos la población modificada
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Reemplazamos la población vieja con la nueva
        population[:] = offspring

        if surrogated is not None:
            model_data += population
            # print(len(model_data))
            if (len(model_data) >= train_size) and (not enough):
                enough = True
                print(f"{fitness_f.__name__} cargando surrogado...")

                # entrenar el modelo
                model, substitutes = train_surrogated(model_data, surrogated, random_seed)

                for part, sub in zip(population, substitutes):
                    part.surrogated = sub
                    # print(part.surrogated )
                # toolbox.register("surrogated", model.predict)

            if enough:
                test = np.array([[p[0], p[1]] for p in population])
                substitutes = model.predict(test)
                for part, sub in zip(population, substitutes):
                    part.surrogated = sub
                    # print(part.surrogated )

        animation_data.append(np.array(population))

        record = mstats.compile(population)
        logbook.record(gen=i + 1, evals=population_size, **record)

    # Obtenemos el mejor individuo de la última población
    # best_ind = tools.selBest(population, 1)[0]
    # print("Mejor solución encontrada: x = %f, y = %f, f(x,y) = %f" % (best_ind[0], best_ind[1], best_ind.fitness.values[0]))

    return population, animation_data, logbook