from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from initToolkit import get_fitness_params


def objective_study(singleObj, loop):
    data = {"data":[], "plot":[], "log":[], "fitness": []}
    for fitness in singleObj:
        #print(fitness.__name__)
        pplt, animation_data, log = loop(n_generations=100, population_size=10,
                                              fitness_f=fitness,
                                              restart_tol=0.01,
                                              train_size=60,
                                              surrogated=(Pipeline([("MMS", MinMaxScaler()), ("SVR", SVR(max_iter=10000))]),
                                                        {'SVR__C': [1, 5, 10],
                                                         'SVR__kernel': ('linear', 'rbf')}),
                                              fitness_params=get_fitness_params()
                                         )
        data["data"].append(pplt)
        data["plot"].append(animation_data)
        data["log"].append(log)
        data["fitness"].append(fitness.__name__)

    return data


def size_study(singleObj, loop, sizes=range(10, 41, 10)):
    dataList = {}
    for fitness in singleObj:
        data = {"data": [], "plot": [], "log": [], "size": []}
        for size in sizes:
            pplt, animation_data, log = loop(n_generations=100,
                                             population_size=size,
                                             fitness_f=fitness,
                                             fitness_params=get_fitness_params()
                                             )

            data["data"].append(pplt)
            data["plot"].append(animation_data)
            data["log"].append(log)
            data["size"].append(size)

        dataList[fitness.__name__] = data

    return dataList


def stab_study(singleObj, loop, seeds=range(174, 1140, 97)):
    dataList = {}
    for fitness in singleObj:
        data = {"data": [], "plot": [], "log": [], "seed": []}
        for seed in seeds:
            pplt, animation_data, log = loop(n_generations=100,
                                             population_size=10,
                                             fitness_f=fitness,
                                             random_seed=seed,
                                             fitness_params=get_fitness_params()
                                             )

            data["data"].append(pplt)
            data["plot"].append(animation_data)
            data["log"].append(log)
            data["seed"].append(seed)
        dataList[fitness.__name__] = data

    return dataList



def train_study(singleObj, loop, sizes=range(10, 121, 30)):
    dataList = {}
    for fitness in singleObj:
        data = {"data": [], "plot": [], "log": [], "train": []}
        for size in sizes:
            pplt, animation_data, log = loop(n_generations=100, population_size=10,
                                             fitness_f=fitness,
                                             restart_tol=0.01,
                                             train_size=size,
                                             surrogated=(
                                             Pipeline([("MMS", MinMaxScaler()), ("SVR", SVR(max_iter=10000))]),
                                             {'SVR__C': [1, 5, 10],
                                              'SVR__kernel': ('linear', 'rbf')}),
                                             fitness_params=get_fitness_params()
                                             )
            data["data"].append(pplt)
            data["plot"].append(animation_data)
            data["log"].append(log)
            data["train"].append(size)

        dataList[fitness.__name__] = data

    return dataList