"""
Archivo que dividiremos por fases lo realizado con la base de datos
"""
#%%
"""
Fase 1. ¿Cuál es el objetivo?¿Queremos predecir o estimar? En nuestro caso al tener ya la base
de datos del repositorio, podemos decir que el objetivo es predecir. La tarea que supone dicha
base de datos es una tarea predictiva, ya que pretendemos estimar el valor de los movimientos 
para ganar, es decir, es tarea de clasificación.
"""
#%%
"""
Fase 2. Recogida de datos-> [[RELLENAR CON LA INFO DE LA BASE DE DATOS]] 
"""
#%%
"""
Fase 3. Preparar y limpiar los datos de entrada.
"""
import matplotlib.pyplot as plt
import pandas as pd

#Lectura de los datos
dataChess = pd.read_csv('krkopt.data',header=',')