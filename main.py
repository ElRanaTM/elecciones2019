# necesita xlrd, usar el comando 'pip install xlrd'

from time import time

# import numpy as np
import pandas as pd
# import matplotlib
from sklearn import linear_model
from sklearn.model_selection import train_test_split

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
