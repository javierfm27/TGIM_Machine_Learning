"""
Archivo que dividiremos por fases con lo realizado en la base de datos
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
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#1.Lectura de los datos
dataChess = pd.read_csv('krkopt.txt',sep=',')

#Le damos unas etiquetas a nuestras características de los datos, ya que carecen de ellas
namesFeatures =['White King Column','White King Row','White Rook Column','White Rook Row' ,
                'Black King Column','Black King Row','Optimal Moves(y)']
dataChess = pd.read_csv('krkopt.txt',sep=',',names=namesFeatures)
dataChess = dataChess.sample(frac=1).reset_index(drop=True) #Baraja los datos

#2.Descripción de los datos
dataChess.describe(include='all')  #En la salida podemos observar como suprime las caracteristicas con la información de la 
#fila en la que se encuentranlas fichas.

#3. MissingValues

#print(dataChess.head(20))
#print("None values: " + str((dataChess['White King File'] == None).sum()))
#print("None values: " + str((dataChess['White King Rank'] == None).sum()))
#print("None values: " + str((dataChess['White Rook File'] == None).sum()))
#print("None values: " + str((dataChess['White Rook Rank'] == None).sum()))

#4. OneHotEncoder
#dataa = ['pepe', 'pepa', 'abelino']
#vals = np.array(dataa)
le = LabelEncoder()
data = dataChess['White King Column'].tolist()
values = np.array(data)
integer_encoded = le.fit_transform(values)
#Binary Encode, crea tres variables binarias a partir de nuestras categorías
enc = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
oh_enc = enc.fit_transform(integer_encoded)
print(oh_enc)

newDataframe = dataChess
newDataFrame[['White King Column']] = oh_enc

#5. Replace Values
arrayValuesY = dataChess['Optimal Moves(y)'].unique()
arrayValuesY = np.array(arrayValuesY)

