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
#%% Representación de los datos
#Posición de rey blanca
# Gráfico de tarta de las variables
plt.subplot(321)
dataChess['White_King_Column'].value_counts().plot(kind='pie', autopct='%.2f',
                                            figsize=(6, 6),
                                            title='Columna de Rey Blanco')
plt.subplot(322)
dataChess['White_King_Row'].value_counts().plot(kind='pie', autopct='%.2f',
                                            figsize=(6, 6),
                                            title='Fila de Rey Blanco')
plt.subplot(323)
dataChess['White_Rook_Column'].value_counts().plot(kind='pie', autopct='%.2f',
                                            figsize=(6, 6),
                                            title='Columna de Torre Blanca')
plt.subplot(324)
dataChess['White_Rook_Row'].value_counts().plot(kind='pie', autopct='%.2f',
                                            figsize=(6, 6),
                                            title='Fila de Torre Blanca')
plt.subplot(325)
dataChess['Black_King_Column'].value_counts().plot(kind='pie', autopct='%.2f',
                                            figsize=(6, 6),
                                            title='Columna de Rey Negro')
plt.subplot(326)
dataChess['Black_King_Row'].value_counts().plot(kind='pie', autopct='%.2f',
                                            figsize=(6, 6),
                                            title='Fila de Rey Negro')
plt.figure()
dataChess['Movements'].value_counts().plot(kind='pie', autopct='%.2f',
                                            figsize=(6, 6),
                                            title='Movimientos para ganar')
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
dataChess.hist()

#%%
"""
Fase 5.Selección del modelo más adecuado y entrenamiento del algoritmo de aprendizaje.
"""
#Debido a que los datos no estan balanceados vamos a eliminar clases en nuestros datos
#newDataChess = newDataChess[newDataChess.Movements.isin(['fourteen','thirteen','twelve','eleven','draw','fitenn','ten','nine','eight'])]
newDataChess = newDataChess[newDataChess.Movements.isin(['fourteen','thirteen'])]


#%%
#Dividimos los datos en target y características
newKeys = newDataChess.keys()
X = newDataChess[newKeys[:-1]].values
y = newDataChess['Movements'].values

X_train, X_test, y_train, y_test = train_test_split(X,y)
#%% NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#Creamos el modelo
model_NB = GaussianNB()

#Entrenamos nuestro modelo
model_NB.fit(X_train,y_train)

#Predecimos las muestras de test
y_hat_NB = model_NB.predict(X_test)
acc_test = accuracy_score(y_test, y_hat_NB)

y_hat_NB2 = model_NB.predict(X_train)
acc_train = accuracy_score(y_train, y_hat_NB2)

#Calculamos el accuracy
print("Accuracy test: " + str(acc_test))
print("Accuracy train: " + str(acc_train))
#%% KNN
from sklearn.neighbors import KNeighborsClassifier
#Creamos el modelo
model_KNN = KNeighborsClassifier(n_neighbors=8)

#Entrenamos nuestro modelo
model_KNN.fit(X_train,y_train)

#Predecimos las muestras de test
y_hat_KNN = model_KNN.predict(X_test)
acc_test = accuracy_score(y_test, y_hat_KNN)

y_hat_KNN2 = model_KNN.predict(X_train)
acc_train = accuracy_score(y_train, y_hat_KNN2)

#Calculamos el accuracy
print("Accuracy test: " + str(acc_test))
print("Accuracy train: " + str(acc_train))
#%% Redes Neuronales
from sklearn.neural_network import MLPClassifier

model_RN = MLPClassifier(max_iter=1000,alpha=1e-8,hidden_layer_sizes=[3,3])

model_RN.fit(X_train, y_train)

y_hat_RN = model_RN.predict(X_test)
acc_test = accuracy_score(y_test, y_hat_RN)

y_hat_RN2 = model_RN.predict(X_train)
acc_train = accuracy_score(y_train, y_hat_RN2)

print("Accuracy test: " + str(acc_test))
print("Accuracy train: " + str(acc_train))
