import os
import pickle
import random
from deap import base, creator, tools
from deap.benchmarks import *
from plotToolkit import *
from GSA import *
from PSO import *
from GA import *
from studies import *


def create_result_dir():
    main = os.getcwd()
    mainpath = os.path.join(main, "resultados")
    mainpp = os.path.join(mainpath, "primera parte")
    mainsp = os.path.join(mainpath, "segunda parte")

    if not os.path.exists(mainpath):
        os.mkdir(mainpath)

    if not os.path.exists(mainpp):
        os.mkdir(mainpp)

    if not os.path.exists(mainsp):
        os.mkdir(mainsp)

    return mainpp, mainsp


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
creator.create("Particle", base=list, fitness=creator.FitnessMin, surrogated=int)

singleObj = [cigar, plane, sphere,
             rand, ackley, bohachevsky,
             griewank, h1,  himmelblau,
             rastrigin, rastrigin_scaled, rastrigin_skew,
             rosenbrock, schaffer, schwefel,
             shekel]

mainpp, mainsp = create_result_dir()
"""
# PRIMERA PARTE: basico
print("INICIO PRIMERA PARTE: BASICO")
print("GSA")
gsadata = objective_study(singleObj, GSA_loop)
print("PSO")
psodata = objective_study(singleObj, PSO_loop)
print("GA")
ga_data = objective_study(singleObj, GA_loop)

print("ESCRITURA")
with open(os.path.join(mainpp, 'gsa_basico.pkl'), 'wb') as archivo:
    pickle.dump(gsadata, file=archivo)

with open(os.path.join(mainpp, 'pso_basico.pkl'), 'wb') as archivo:
    pickle.dump(psodata, file=archivo)

with open(os.path.join(mainpp, 'ga_basico.pkl'), 'wb') as archivo:
    pickle.dump(ga_data, file=archivo)


# PRIMERA PARTE: size
print("INICIO PRIMERA PARTE: SIZE")
print("GSA")
gsadata = size_study(singleObj, GSA_loop)
print("PSO")
psodata = size_study(singleObj, PSO_loop)
print("GA")
ga_data = size_study(singleObj, GA_loop)

print("ESCRITURA")
with open(os.path.join(mainpp, 'gsa_size.pkl'), 'wb') as archivo:
    pickle.dump(gsadata, file=archivo)

with open(os.path.join(mainpp, 'pso_size.pkl'), 'wb') as archivo:
    pickle.dump(psodata, file=archivo)

with open(os.path.join(mainpp, 'ga_size.pkl'), 'wb') as archivo:
    pickle.dump(ga_data, file=archivo)


# PRIMERA PARTE: stab
print("INICIO PRIMERA PARTE: STAB")
print("GSA")
gsadata = stab_study(singleObj, GSA_loop)
print("PSO")
psodata = stab_study(singleObj, PSO_loop)
print("GA")
ga_data = stab_study(singleObj, GA_loop)

print("ESCRITURA")
with open(os.path.join(mainpp, 'gsa_stab.pkl'), 'wb') as archivo:
    pickle.dump(gsadata, file=archivo)

with open(os.path.join(mainpp, 'pso_stab.pkl'), 'wb') as archivo:
    pickle.dump(psodata, file=archivo)

with open(os.path.join(mainpp, 'ga_stab.pkl'), 'wb') as archivo:
    pickle.dump(ga_data, file=archivo)
"""

# SEGUNDA PARTE: trainset
print("INICIO SEGUNDA PARTE")
print("GSA")
sizes_list = [100, 200, 300, 400]
gsadata = train_study(singleObj, GSA_loop, sizes=sizes_list)
print("PSO")
psodata = train_study(singleObj, PSO_loop, sizes=sizes_list)
print("GA")
ga_data = train_study(singleObj, GA_loop, sizes=sizes_list)

print("ESCRITURA")
with open(os.path.join(mainsp, 'gsa_trainset.pkl'), 'wb') as archivo:
    pickle.dump(gsadata, file=archivo)

with open(os.path.join(mainsp, 'pso_trainset.pkl'), 'wb') as archivo:
    pickle.dump(psodata, file=archivo)

with open(os.path.join(mainsp, 'ga_trainset.pkl'), 'wb') as archivo:
    pickle.dump(ga_data, file=archivo)