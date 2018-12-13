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
from sklearn.model_selection import train_test_split

#1.Lectura de los datos
dataChess = pd.read_csv('krkopt.txt',sep=',')

#Le damos unas etiquetas a nuestras características de los datos, ya que carecen de ellas
namesFeatures =['White_King_Column','White_King_Row','White_Rook_Column','White_Rook_Row' ,
                'Black_King_Column','Black_King_Row','Movements']
dataChess = pd.read_csv('krkopt.txt',sep=',',names=namesFeatures)
dataChess = dataChess.sample(frac=1).reset_index(drop=True) #Baraja los datos
#%%
#2.Descripción de los datos
dataChess.describe(include='all')  #En la salida podemos observar como suprime las caracteristicas con la información de la 
#fila en la que se encuentranlas fichas.
#%%
#3. MissingValues
#Una de las manneras para ver si tenemos o no missing values es comprobar el número de ocurrencias de cada valor.
for i in namesFeatures:
    print(dataChess[i].value_counts())
#%% Categorizar variables
#4. OneHotEncoder
    
newDataChess = pd.DataFrame()
for feature in namesFeatures[:-1]:
    #feature_position = dataChess.columns.get_loc(feature)
    newCat = pd.get_dummies(dataChess[feature])
    
    newKeys = []
    for key in newCat.keys():
        newKey = feature + "_" + str(key)
        newKeys.append(newKey)
        
    newCat.columns = newKeys
    
    newDataChess = pd.concat([newDataChess, newCat], axis = 1, sort = False)
    #newDataChess.insert(feature_position, newKeys, newCat)

newDataChess = pd.concat([newDataChess, dataChess['Movements']], axis = 1, sort = False)

#category_White_kpd.get_dummies(dataChess['White King Column')
#category_White_King_Column.columns = ['WhiteKingColumn_A','WhiteKingColumn_B','WhiteKingColumn_C','WhiteKingColumn_D']
#%%
newDataChess['White_King_Column_a'].hist()

#%% 
"""
Fase 5.Selección del modelo más adecuado y entrenamiento del algoritmo de aprendizaje.
"""
#Debido a que los datos no estan balanceados vamos a eliminar clases en nuestros datos
newDataChess = newDataChess[newDataChess.Movements.isin(['fourteen','thirteen','twelve','eleven','draw','fitenn','ten','nine','eight'])]
#newDataChess = newDataChess[newDataChess.Movements.isin(['fourteen','thirteen'])]


#%%
#Dividimos los datos en target y características
newKeys = newDataChess.keys()
X = newDataChess[newKeys[:-1]].values
y = newDataChess['Movements'].values

X_train, X_test, y_train, y_test = train_test_split(X,y)
#%% NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
#Creamos el modelo
model_NB = GaussianNB()

#Entrenamos nuestro modelo
y_pred = model_NB.fit(X_train,y_train)

#Predecimos las muestras de test 
y_hat_NB = model_NB.predict(X_test)

#Calculamos el accuracy
acc = np.mean(y_test == y_hat_NB)
print(acc)
#%% KNN
from sklearn.neighbors import KNeighborsClassifier
#Creamos el modelo
model_KNN = KNeighborsClassifier(n_neighbors=6)

#Entrenamos nuestro modelo
y_pred = model_KNN.fit(X_train,y_train)

#Predecimos las muestras de test
y_hat_modelKNN = model_KNN.predict(X_test)

#Accuracy
acc = np.mean(y_test == y_hat_modelKNN)
print(acc)
#%% Redes Neuronales
from sklearn.neural_network import MLPClassifier

model_RN = MLPClassifier(max_iter=1000,alpha=1e-8)

model_RN.fit(X_train, y_train) 
y_hat_RN = model_RN.predict(X_test)
acc = np.mean(y_hat_RN == y_test)
print(acc)