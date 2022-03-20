# ==================================================================
# Universidade Estadual de Feira de Santana
# Mestrado em Ciência da Computação
# Disciplina: PGCC015 - Inteligência Computacional
# Professor: Matheus Giovanni Pires
# Aluno: Noberto Pires Maciel
# EPC07 - 10/12/2020
# Algoritmo genético - problema da antena acoplada
# lib: pymoo
# ==================================================================

# import general libs
import math
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import pymoo
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.optimize import minimize
from pymoo.interface import mutation,crossover
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_selection, get_problem, get_termination, get_sampling, get_crossover, get_mutation
from pymoo.visualization.scatter import Scatter
from pymoo.model.problem import Problem

x = []
y = []

def sin(x):
  for i in range(len(x)):
    y.append(x[i]*math.sin(10*math.pi*x[i])+1)


class AntenaAcoplada(Problem):
    def __init__(self,**kwargs):
        super().__init__(n_var=1, n_obj=1, n_constr=0, xl=-1, xu=2, elementwise_evaluation=True, **kwargs)
    def _evaluate(self, x, out, *args, **kwargs):
        out["X"] = x
        out["F"] = -1*(x*math.sin(10*math.pi*x)+1)

problem = AntenaAcoplada()

# roleta
def weighted_random_choice(choices):
    max = sum(choices.values())
    pick = random.uniform(0, max)
    current = 0
    for key, value in choices.items():
        current += value
        if current > pick:
            return key


selection = get_selection('tournament', {'pressure' : 2, 'func_comp' : weighted_random_choice})

def run(cross=0.7,ag=None,mut=0.01):
    algorithm = GA(pop_size=100,
                      sampling=get_sampling("real_random"),
                      crossover=get_crossover("real_one_point",prob=cross),
                      mutation=get_mutation("real_pm", eta=20, prob=mut), #bin_bitflip
                      n_offsprings=ag, #ag=1 -> steady-state, ag=None -> geracional
                      eliminate_duplicates=True)

    X,F = problem.evaluate(np.random.rand(100, 1),return_values_of=["X","F"])

    termination = get_termination("n_eval", 25000)

    res = minimize(problem,
                  algorithm,
                  ('n_gen', 200),
                  termination,
                  seed=10, # elitismo
                  verbose=False)
    return X,F,res

# questão 1 AG: cruzamento em 70%, 80% e 90%. Taxa de mutação de 1%. Taxa  do  elitismo  10%
# questão 2 ST: cruzamento em 70%, 80% e 90%. Taxa de mutação de 1%. Taxa  do  elitismo  10%
bestCross = []
for j in [0.7,0.8,0.9]:
    X,F,res = run(cross=j,ag=None,mut=0.1)
    for i in np.arange(-1,2.01,0.01):
      x.append(i)
    sin(x)
    plt.plot(x,y, label="x*sen(10*pi*x)+1")
    plt.scatter(res.X,res.F*(-1), color="red",marker="x")
    print('Item 1 EPC07 Geracional: maximo global de f(x):',res.F*(-1),' x:', res.X,' Cross tx: ',j*100,'%')
    #print('Item 2 EPC07 Steady-State: maximo global de f(x):',res.F*(-1),' x:', res.X,' Cross tx: ',j*100,'%')

    bestCross.append((res.F*(-1),j*100))
    plt.scatter(X,F*(-1), color="blue", alpha=0.2)
    plt.xlabel("Item 1 EPC07 Geracional: máximo global f(x)")
    #plt.xlabel("Item 2 EPC07 Steady-State: máximo global f(x)")
    plt.ylabel("y")
    plt.legend(loc='upper left')
    plt.show()
    x = []
    y = []

# questão 3 AG: mutação  em  1%,  5%  e  10%. Taxa  do elitismo  10%. Taxa de cruzamento?
# questão 4 ST: mutação  em  1%,  5%  e  10%. Taxa  do elitismo  10%. Taxa de cruzamento?
for j in [0.01,0.05,0.1]:
    X,F,res = run(cross=max(bestCross)[1],ag=None,mut=j)
    for i in np.arange(-1,2.01,0.01):
      x.append(i)
    sin(x)
    plt.plot(x,y, label="x*sen(10*pi*x)+1")
    plt.scatter(res.X,res.F*(-1), color="red",marker="x")
    print('Item 3 EPC07 Geracional: maximo global de f(x):',res.F*(-1),' x:', res.X,' Cross tx: ',max(bestCross)[1],'% Mutation tx: ',j*100,'%')
    #print('Item 4 EPC07 Steady-State: maximo global de f(x):',res.F*(-1),' x:', res.X,' Cross tx: ',max(bestCross)[1],'% Mutation tx: ',j*100,'%')

    plt.scatter(X,F*(-1), color="blue", alpha=0.2)
    plt.xlabel("Item 3 EPC07 Geracional: máximo global f(x)")
    #plt.xlabel("Item 4 EPC07 Steady-State: máximo global f(x)")
    plt.ylabel("y")
    plt.legend(loc='upper left')
    plt.show()
    x = []
    y = []
