import numpy as np
import matplotlib.pyplot as plt
from deap.benchmarks import shekel
import os
import pickle
import random
from deap import base, creator, tools
import math
from deap.benchmarks import *
from plotToolkit import *
from initToolkit import get_fitness_params
#creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#creator.create("Individual", list, fitness=creator.FitnessMin)
#creator.create("Particle", base=list, fitness=creator.FitnessMin, surrogated=int)

def calculate_margins(points, margin):
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])

    xlen = xmax - xmin
    ylen = ymax - ymin

    if xlen < 1.e-15:
        xlen = 1

    if ylen < 1.e-15:
        ylen = 1

    xmin = xmin - (margin * xlen)
    xmax = xmax + (margin * xlen)
    ymin = ymin - (margin * ylen)
    ymax = ymax + (margin * ylen)

    return [xmin, xmax], [ymin, ymax]




def plot_mean_and_std(log, chapter, titulo=None, fig=None, ax=None, legend=False):

    gen = log.chapters[chapter].select("gen")
    mean = log.chapters[chapter].select("avg")
    std = log.chapters[chapter].select("std")

    # Comprueba que los tres parámetros son de tipo numpy.ndarray
    gen = np.array(gen) - 1
    mean = np.array(mean)
    std = np.array(std)

    # Usa los objetos de figura y ejes existentes o crea unos nuevos si no se proporcionan
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # Itera sobre los puntos y agrega las líneas verticales para las desviaciones estándar
    lines = []
    for i in gen:
        line, = ax.plot([i, i], [mean[i] -std[i], mean[i] +std[i]], color='#2C3E50', label="Desviación")
        lines.append(line)

    # Agrega los puntos de dispersión
    line = ax.scatter(gen, mean, s=10, color='#FF0033', zorder=2, label='Media')
    lines.append(line)

    # Configura los ejes x e y
    ax.set_xlabel('Generación')
    ax.set_ylabel('Función Objetivo')


    # Calcula los límites del gráfico
    xmin = min(gen) - 2.5
    xmax = max(gen) + 2.5
    ymin = min(mean - std) - 3
    ymax = max(mean + std) + 3
    plt.axis([xmin, xmax, ymin, ymax])

    if legend:
        ax.legend(lines[-2:], ['Desviación estándar', 'Media'], loc='upper right')


    # Agrega el título si se proporciona
    if titulo:
        ax.set_title(titulo)

    return fig, ax


def plot_fit_sub(log, titulo=None, legend=False, fig=None, ax=None):
    surroga = log.chapters["surrogated"].select("avg")
    fitness = log.chapters["fitness"].select("avg")

    surroga = np.array(surroga)
    fitness = np.array(fitness)

    # Usa los objetos de figura y ejes existentes o crea unos nuevos si no se proporcionan
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.plot(surroga, label="surrogated")
    ax.plot(fitness, label="fitness")

    # Configura los ejes x e y
    ax.set_xlabel('Generación')
    ax.set_ylabel('Media')

    # Calcula los límites del gráfico
    """
    xmin = min(gen) - 2.5
    xmax = max(gen) + 2.5
    ymin = min([np.nanmin(surroga), np.nanmin(fitness)]) - 3
    ymax = min([np.nanmax(surroga), np.nanmax(fitness)]) + 3
    plt.axis([xmin, xmax, ymin, ymax])
    """

    if legend:
        ax.legend()

    # Agrega el título si se proporciona
    if titulo:
        ax.set_title(titulo)

    return fig, ax


#plot_fit_sub(log, "fitness", "data")

def contour_plot(function, xlim, ylim, xbin, ybin, points=None, fig=None, ax=None, title=None):

    # Definir el rango del grafico, se centra en (0,0)
    xmin, xmax = xlim
    ymin, ymax = ylim
    xlist = np.linspace(xmin, xmax, xbin)
    ylist = np.linspace(ymin, ymax, ybin)

    # Definir la malla de puntos
    X, Y = np.meshgrid(xlist, ylist)

    # Guardar el valor de la funcion para cada punto
    z = []
    for i, j in zip(X.ravel(), Y.ravel()):
        z.append(function([i, j]))

    # Pasar a las mismas dimensiones que X e Y
    Z = np.array(z).reshape((xbin, ybin))

    # Mostrar el grafico
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)

    cp = ax.contourf(X, Y, Z, cmap='jet', alpha=0.4)

    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], color="black", s=10)

    fig.colorbar(cp, ax=ax)  # Add a colorbar to a plot
    if title is None:
        ax.set_title(f'{function.__name__.capitalize()} Contour')
    else:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return fig, ax


def size_plot(data, fit, stat="avg", fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)

    for i in range(len(data[fit]["log"])):
        gen = data[fit]["log"][i].chapters["fitness"].select("gen")
        size = data[fit]["log"][i].chapters["fitness"].select("evals")[0]

        if "avg".__eq__(stat):
            mean = data[fit]["log"][i].chapters["fitness"].select("avg")
            ax.plot(gen, mean, label=f"{size} ind")

        if "min".__eq__(stat):
            _min = data[fit]["log"][i].chapters["fitness"].select("min")
            _max = data[fit]["log"][i].chapters["fitness"].select("max")
            ax.plot(gen, _min, label=f"{size} individuos")
            ax.plot(gen, _max, label=f"{size} individuos")

        #ax.legend()
        ax.set_title(fit)

    return fig, ax


def stab_plot(data, fit, stat="avg", fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)

    seedlist = data[fit]['seed']

    for i in range(len(data[fit]['seed'])):
        seed = seedlist[i]
        gen = data[fit]["log"][i].chapters["fitness"].select("gen")

        if "avg".__eq__(stat):
            mean = data[fit]["log"][i].chapters["fitness"].select("avg")
            ax.plot(gen, mean, label=f"semilla {seed}")

        if "min".__eq__(stat):
            _min = data[fit]["log"][i].chapters["fitness"].select("min")
            _max = data[fit]["log"][i].chapters["fitness"].select("max")
            ax.plot(gen, _min, label=f"semilla {seed}")
            ax.plot(gen, _max, label=f"semilla {seed}")

        # ax.legend()
        ax.set_title(fit)

    return fig, ax


def train_plot(data, fitness, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)

    trainset = data[fitness]["train"]
    gen = data[fitness]["log"][0].chapters["fitness"].select("gen")
    fit = data[fitness]["log"][0].chapters["fitness"].select("avg")
    ax.plot(gen, fit, label="fitness")

    for i, train in enumerate(trainset):
        sur = data[fitness]["log"][i].chapters["surrogated"].select("avg")
        ax.plot(gen, sur, label=f"train: {train}")
    #ax.legend()
    ax.set_title(fitness)

    return fig, ax

def fit_dev(data):
    # Define el número de subplots y sus posiciones
    nrows = 4
    ncols = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))

    # Itera sobre los subplots y agrega la gráfica correspondiente
    i = 0
    for row in range(nrows):
        for col in range(ncols):
            ax = axs[row, col]
            if i < len(data["log"]):
                log = data["log"][i]
                fit = data["fitness"][i]
                fig, ax = plot_mean_and_std(log, "fitness", fit, fig=fig, ax=ax)
                axs[row, col] = ax
            else:
                ax.axis("off")
            i += 1

    fig.tight_layout()
    return fig, ax

def contour_grid(data, singleObj, margin=0.05, xbin=500, ybin=500, fitness_params = get_fitness_params()):

    nrows = 4
    ncols = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))

    i = 0
    for row in range(nrows):
        for col in range(ncols):
            ax = axs[row, col]
            if i < len(singleObj):
                points = np.array(data["data"][i])
                xlim, ylim = calculate_margins(points, margin)
                fitness = singleObj[i]
                #print(i, fitness.__name__)
                if 'shekel'.__eq__(fitness.__name__):
                    fitness = fitness_params["shekel"]
                elif "h1".__eq__(fitness.__name__):
                    fitness = fitness_params["h1"]

                fig, ax = contour_plot(fitness, xlim, ylim, xbin, ybin, points=points, fig=fig, ax=ax)
                axs[row, col] = ax
            else:
                ax.axis("off")
            i += 1

    fig.tight_layout()
    return fig, ax



def fit_vs_sur(data):
    # Define el número de subplots y sus posiciones
    nrows = 4
    ncols = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))

    # Itera sobre los subplots y agrega la gráfica correspondiente
    i = 0
    for row in range(nrows):
        for col in range(ncols):
            ax = axs[row, col]
            if i < len(data["log"]):
                log = data["log"][i]
                fit = data["fitness"][i]
                fig, ax = plot_fit_sub(log, fit, fig=fig, ax=ax, legend=True)
                axs[row, col] = ax
            else:
                ax.axis("off")
            i += 1

    handles, labels = ax.get_legend_handles_labels()
    num_cols = math.ceil(len(labels))
    fig.legend(handles, labels, loc='upper center', ncol=num_cols, bbox_to_anchor=(0.5, 1.025))
    fig.tight_layout()
    return fig, ax


def size_grid(data):
    # Define el número de subplots y sus posiciones
    nrows = 4
    ncols = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    fit_names = list(data.keys())
    # Itera sobre los subplots y agrega la gráfica correspondiente
    i = 0
    for row in range(nrows):
        for col in range(ncols):
            ax = axs[row, col]
            if i < len(fit_names):
                fit = fit_names[i]
                fig, ax = size_plot(data, fit, "avg", fig, ax)
                axs[row, col] = ax
            else:
                ax.axis("off")
            i += 1

    handles, labels = ax.get_legend_handles_labels()
    num_cols = math.ceil(len(labels))
    fig.legend(handles, labels, loc='upper center', ncol=num_cols, bbox_to_anchor=(0.5, 1.025))
    fig.tight_layout()
    return fig, ax



def stab_grid(data):
    # Define el número de subplots y sus posiciones
    nrows = 4
    ncols = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    fit_names = list(data.keys())
    # Itera sobre los subplots y agrega la gráfica correspondiente
    i = 0
    for row in range(nrows):
        for col in range(ncols):
            ax = axs[row, col]
            if i < len(fit_names):
                fit = fit_names[i]
                fig, ax = stab_plot(data, fit, "avg", fig, ax)
                axs[row, col] = ax
            else:
                ax.axis("off")
            i += 1

    handles, labels = ax.get_legend_handles_labels()

    if math.ceil(len(labels))==10:
        num_cols = 5
    else:
        num_cols = math.ceil(len(labels))

    fig.legend(handles, labels, loc='upper center', ncol=num_cols, bbox_to_anchor=(0.5, 1.045))
    fig.tight_layout()
    return fig, ax


def train_grid(data):
    # Define el número de subplots y sus posiciones
    nrows = 4
    ncols = 4
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    fit_names = list(data.keys())
    # Itera sobre los subplots y agrega la gráfica correspondiente
    i = 0
    for row in range(nrows):
        for col in range(ncols):
            ax = axs[row, col]
            if i < len(fit_names):
                fit = fit_names[i]
                fig, ax = train_plot(data, fit, fig, ax)
                axs[row, col] = ax
            else:
                ax.axis("off")
            i += 1

    handles, labels = ax.get_legend_handles_labels()
    num_cols = math.ceil(len(labels))
    fig.legend(handles, labels, loc='upper center', ncol=num_cols, bbox_to_anchor=(0.5, 1.025))
    fig.tight_layout()
    return fig, ax


# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center')










