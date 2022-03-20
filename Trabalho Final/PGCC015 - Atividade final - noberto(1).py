# ==================================================================
# Universidade Estadual de Feira de Santana
# Mestrado em Ciência da Computação
# Disciplina: PGCC015 - Inteligência Computacional
# Professor: Matheus Giovanni Pires
# Aluno: Noberto Pires Maciel
# Atividade final - 14/12/2020
# Métricas de qualidade para o sistema de transporte público urbano
# para a perspectiva do passageiro através de dados gerais
# ==================================================================

# questions:
# Quais dados dos sistemas utilizar para aferir a qualidade do transporte para o passageiro?
# Todos os dados são definidos diariamente com base nos veículos que entram e saem

# Veículos por habitante
# Número de linhas por bairro
# Número de horários por linha
# Tempo de espera por ponto de embarque e desembarque

    #Leitura dos atributos  do dataset no espaço (range)
    #@attribute VehiclesPerMilHabitant real [102, 738]
    #@attribute HabitantsPerBus real [600, 6000]
    #@attribute WaitTime real [5, 120] minutes

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl, interp_membership, interp_universe, defuzz


# 100 pontos de discretização para todos os universos abaixo
x_VehiclesPerMilHabitant = np.arange(102, 738, 6.36)
x_HabitantsPerBus = np.arange(600, 6054, 54)
x_WaitTime  = np.arange(5, 121, 1.16)
x_QualityOfService  = np.arange(0, 101, 1)

VehiclesPerMilHabitant = ctrl.Antecedent(x_VehiclesPerMilHabitant, 'VehiclesPerMilHabitant')
HabitantsPerBus = ctrl.Antecedent(x_HabitantsPerBus, 'HabitantsPerBus')
WaitTime = ctrl.Antecedent(x_WaitTime, 'WaitTime')
QualityOfService = ctrl.Consequent(x_QualityOfService, 'QualityOfService')

VehiclesPerMilHabitant['poucos'] = fuzz.trapmf(VehiclesPerMilHabitant.universe, [102, 102, 248.28, 413.64])
VehiclesPerMilHabitant['medio'] = fuzz.trimf(VehiclesPerMilHabitant.universe,  [248.28, 413.64, 579])
VehiclesPerMilHabitant['muitos'] = fuzz.trapmf(VehiclesPerMilHabitant.universe, [413.64, 579, 738, 738])

HabitantsPerBus['poucos'] = fuzz.trapmf(HabitantsPerBus.universe,  [600, 600, 1842, 3246])
HabitantsPerBus['medio'] = fuzz.trimf(HabitantsPerBus.universe,  [1842, 3246, 4650])
HabitantsPerBus['muitos'] = fuzz.trapmf(HabitantsPerBus.universe,  [3246, 4650, 6000, 6000])

WaitTime['pouco'] = fuzz.trapmf(WaitTime.universe,  [5, 5, 31.68, 61.84])
WaitTime['medio'] = fuzz.trimf(WaitTime.universe, [31.68, 61.84, 92])
WaitTime['muito'] = fuzz.trapmf(WaitTime.universe,  [61.84, 92, 121, 121])

QualityOfService['ruim'] = fuzz.trapmf(QualityOfService.universe,  [0, 0, 23, 49])
QualityOfService['razoavel'] = fuzz.trimf(QualityOfService.universe, [23, 49, 75])
QualityOfService['boa'] = fuzz.trapmf(QualityOfService.universe,  [49, 75, 100, 100])

QualityOfService.defuzzify_method = 'centroid'
QualityOfService.accumulation_method = max

VehiclesPerMilHabitant.view()
HabitantsPerBus.view()
WaitTime.view()
QualityOfService.view()

# regras
regra1 = ctrl.Rule(VehiclesPerMilHabitant['poucos'] & HabitantsPerBus['poucos'] & WaitTime['muito'],QualityOfService['ruim'])
regra2 = ctrl.Rule(VehiclesPerMilHabitant['medio'] & HabitantsPerBus['poucos'] & WaitTime['muito'],QualityOfService['ruim'])
regra3 = ctrl.Rule(VehiclesPerMilHabitant['muitos'] & HabitantsPerBus['poucos'] & WaitTime['muito'],QualityOfService['ruim'])
regra4 = ctrl.Rule(VehiclesPerMilHabitant['poucos'] & HabitantsPerBus['medio'] & WaitTime['muito'],QualityOfService['ruim'])
regra5 = ctrl.Rule(VehiclesPerMilHabitant['medio'] & HabitantsPerBus['medio'] & WaitTime['muito'],QualityOfService['ruim'])
regra6 = ctrl.Rule(VehiclesPerMilHabitant['muitos'] & HabitantsPerBus['medio'] & WaitTime['muito'],QualityOfService['ruim'])
regra7 = ctrl.Rule(VehiclesPerMilHabitant['poucos'] & HabitantsPerBus['muitos'] & WaitTime['muito'],QualityOfService['ruim'])
regra8 = ctrl.Rule(VehiclesPerMilHabitant['medio'] & HabitantsPerBus['muitos'] & WaitTime['muito'],QualityOfService['ruim'])
regra9 = ctrl.Rule(VehiclesPerMilHabitant['muitos'] & HabitantsPerBus['muitos'] & WaitTime['muito'],QualityOfService['ruim'])

regra10 = ctrl.Rule(VehiclesPerMilHabitant['poucos'] & HabitantsPerBus['poucos'] & WaitTime['medio'],QualityOfService['razoavel'])
regra11 = ctrl.Rule(VehiclesPerMilHabitant['medio'] & HabitantsPerBus['poucos'] & WaitTime['medio'],QualityOfService['razoavel'])
regra12 = ctrl.Rule(VehiclesPerMilHabitant['muitos'] & HabitantsPerBus['poucos'] & WaitTime['medio'],QualityOfService['razoavel'])
regra13 = ctrl.Rule(VehiclesPerMilHabitant['poucos'] & HabitantsPerBus['medio'] & WaitTime['medio'],QualityOfService['razoavel'])
regra14 = ctrl.Rule(VehiclesPerMilHabitant['medio'] & HabitantsPerBus['medio'] & WaitTime['medio'],QualityOfService['razoavel'])
regra15 = ctrl.Rule(VehiclesPerMilHabitant['muitos'] & HabitantsPerBus['medio'] & WaitTime['medio'],QualityOfService['razoavel'])
regra16 = ctrl.Rule(VehiclesPerMilHabitant['poucos'] & HabitantsPerBus['muitos'] & WaitTime['medio'],QualityOfService['razoavel'])
regra17 = ctrl.Rule(VehiclesPerMilHabitant['medio'] & HabitantsPerBus['muitos'] & WaitTime['medio'],QualityOfService['razoavel'])
regra18 = ctrl.Rule(VehiclesPerMilHabitant['muitos'] & HabitantsPerBus['muitos'] & WaitTime['medio'],QualityOfService['razoavel'])

regra19 = ctrl.Rule(VehiclesPerMilHabitant['poucos'] & HabitantsPerBus['poucos'] & WaitTime['pouco'],QualityOfService['boa'])
regra20 = ctrl.Rule(VehiclesPerMilHabitant['medio'] & HabitantsPerBus['poucos'] & WaitTime['pouco'],QualityOfService['boa'])
regra21 = ctrl.Rule(VehiclesPerMilHabitant['muitos'] & HabitantsPerBus['poucos'] & WaitTime['pouco'],QualityOfService['boa'])
regra22 = ctrl.Rule(VehiclesPerMilHabitant['poucos'] & HabitantsPerBus['medio'] & WaitTime['pouco'],QualityOfService['boa'])
regra23 = ctrl.Rule(VehiclesPerMilHabitant['medio'] & HabitantsPerBus['medio'] & WaitTime['pouco'],QualityOfService['boa'])
regra24 = ctrl.Rule(VehiclesPerMilHabitant['muitos'] & HabitantsPerBus['medio'] & WaitTime['pouco'],QualityOfService['boa'])
regra25 = ctrl.Rule(VehiclesPerMilHabitant['poucos'] & HabitantsPerBus['muitos'] & WaitTime['pouco'],QualityOfService['boa'])
regra26 = ctrl.Rule(VehiclesPerMilHabitant['medio'] & HabitantsPerBus['muitos'] & WaitTime['pouco'],QualityOfService['boa'])
regra27 = ctrl.Rule(VehiclesPerMilHabitant['muitos'] & HabitantsPerBus['muitos'] & WaitTime['pouco'],QualityOfService['boa'])

regra1.view()

quality_ctrl = ctrl.ControlSystem([regra1, regra2,regra3,regra4,regra5,regra6,regra7,regra8,regra9,regra10,regra11,regra12,regra13,regra14,regra15,regra16,regra17,regra18,regra19,regra20,regra21,regra22,regra23,regra24,regra25,regra26,regra27])
quality_sim = ctrl.ControlSystemSimulation(quality_ctrl)


    #@attribute VehiclesPerMilHabitant real [102, 738]
    #@attribute HabitantsPerBus real [600, 6000]
    #@attribute WaitTime real [5, 120] minutes

# Curitiba  PR = [[745],[1399,51],[54]]
# Salvador  BA = [[374],[1338,5],[55]]
# São Paulo SP = [[625],[3974,37],[62]]

#simulate = [[VehiclesPerMilHabitant],[HabitantsPerBus],[WaitTime]]
simulate = [[745,374,625],[1399.51,1338.5,3974.37],[54,55,62]]

for x in range(len(simulate[0])):
  quality_sim.input['VehiclesPerMilHabitant'] = simulate[0][x]
  quality_sim.input['HabitantsPerBus'] = simulate[1][x]
  quality_sim.input['WaitTime'] = simulate[2][x]
  quality_sim.compute()
  print('simulação',x+1,': ',quality_sim.output['QualityOfService'])
  QualityOfService.view(sim=quality_sim)

print(VehiclesPerMilHabitant,simulate[0])
print(HabitantsPerBus,simulate[1])
print(WaitTime,simulate[2])
print(QualityOfService)