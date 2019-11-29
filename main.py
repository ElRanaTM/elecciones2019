# necesita xlrd, usar el comando 'pip install xlrd'

from time import time

import regresion_polinomial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn import linear_model
# from sklearn.model_selection import train_test_split

tiempo_inicial = time()
actas = pd.read_excel('actas.xlsx')
tiempo_final = time()
print("tiempo de importación: ", tiempo_final - tiempo_inicial)
input("Presione ENTER para continuar...")

print("Ejemplos de las 15 primeras filas")
print(actas.head(15))
input("Presione ENTER para continuar...")

print("Existen tres tipos de elección, presidencial, diputados uninominales y diputados especiales,")
print("Al ser todos objetos desconocidos, tomaremos todos los datos del dataset, a continuación un poco de info.")
input("Presione ENTER para continuar...")

print(actas.info())
input("Presione ENTER para continuar...")

print("Definimos X")
input("Presione ENTER para continuar...")
X = actas[['CC', 'FPV', 'MTS', 'MAS - IPSP', '21F', 'PDC', 'MNR', 'PAN-BOL']]
print(X.info())
input("Presione ENTER para continuar...")

print("Definimos y")
input("Presione ENTER para continuar...")
y = actas[['Votos Válidos']]
print(y.info())
input("Presione ENTER para continuar...")

newColumn = np.power(X, 7)
print(newColumn.shape)

X_x2agregado = np.append(X, newColumn, 1)
print(X_x2agregado.shape)

plt.plot(X_x2agregado[:, 0], y, "y.", ms=1)
plt.title("Comunidad Ciudadana")
plt.xlabel('Cabtidad de Votantes')
plt.ylabel('Cantidad de inscritos')
plt.show()
plt.plot(X_x2agregado[:, 1], y, "m.", ms=1)
plt.title("Frente Para la Victoria")
plt.xlabel('Cabtidad de Votantes')
plt.ylabel('Cantidad de inscritos')
plt.show()
plt.plot(X_x2agregado[:, 2], y, "g.", ms=1)
plt.title("Movimiento Tercer Sistema")
plt.xlabel('Cabtidad de Votantes')
plt.ylabel('Cantidad de inscritos')
plt.show()
plt.plot(X_x2agregado[:, 3], y, "b.", ms=1)
plt.title("Movimiento Al Socialismo")
plt.xlabel('Cabtidad de Votantes')
plt.ylabel('Cantidad de inscritos')
plt.show()
plt.plot(X_x2agregado[:, 4], y, "r.", ms=1)
plt.title("Bolivia Dice No 21F")
plt.xlabel('Cabtidad de Votantes')
plt.ylabel('Cantidad de inscritos')
plt.show()
plt.plot(X_x2agregado[:, 5], y, "k.", ms=1)
plt.title("Partido Democrata Cristiano")
plt.xlabel('Cabtidad de Votantes')
plt.ylabel('Cantidad de inscritos')
plt.show()
plt.plot(X_x2agregado[:, 6], y, "c.", ms=1)
plt.title("Movimiento Nacionalista Revolucionario")
plt.xlabel('Cabtidad de Votantes')
plt.ylabel('Cantidad de inscritos')
plt.show()
plt.plot(X_x2agregado[:, 7], y, "r.", ms=1)
plt.title("Partido de Accion Nacional Boliviano")
plt.xlabel('Cabtidad de Votantes')
plt.ylabel('Cantidad de inscritos')
plt.show()
'''
# Entrenamos los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Definimos el algoritmo a utilizar
lr_multiple = linear_model.LinearRegression()

# Entrenamos el modelo
lr_multiple.fit(X_train, y_train)

# Realizamos una predicción
Y_pred_multiple = lr_multiple.predict(X_test)

print('DATOS DEL MODELO REGRESIÓN LINEAL MULTIPLE')
print()
print('Valor de las pendientes o coeficientes "a":')
print(lr_multiple.coef_)
print('Valor de la intersección o coeficiente "b":')
print(lr_multiple.intercept_)
input("Presione ENTER para continuar...")

print('Precisión del modelo:')
print(lr_multiple.score(X_train, y_train))
'''
