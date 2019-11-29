# Archivo principal
# necesita xlrd, usar el comando 'pip install xlrd'

from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regresion_polinomial as rp
import graficos as gf

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
# print(newColumn.shape)

X_x2agregado = np.append(X, newColumn, 1)
# print(X_x2agregado.shape)

band = True
while band:
    print("Votantes vs Inscritos")
    print("Elija un partido")
    print("1. Comunidad Ciudadana")
    print("2. Frente Para la Victoria")
    print("3. Movimiento Tercer Sistema")
    print("4. Movimiento Al Socialismo")
    print("5. Bolivia Dice No 21F")
    print("6. Partido Democrata Cristiano")
    print("7. Movimiento Nacionalista Revolucionario")
    print("8. Partido de Accion Nacional Boliviano")
    print("9. Continuar")
    opc = input("ingrese una opcion valida(1-9): ")
    if opc == "1":
        gf.imprimirCC(X_x2agregado, y)
    if opc == "2":
        gf.imprimirFPV(X_x2agregado, y)
    if opc == "3":
        gf.imprimirMTS(X_x2agregado, y)
    if opc == "4":
        gf.imprimirMAS(X_x2agregado, y)
    if opc == "5":
        gf.imprimir21F(X_x2agregado, y)
    if opc == "6":
        gf.imprimirPDC(X_x2agregado, y)
    if opc == "7":
        gf.imprimirMNR(X_x2agregado, y)
    if opc == "8":
        gf.imprimirPAN(X_x2agregado, y)
    if opc == "9":
        band = False

X_norm, mu, sigma = rp.normalizarCaracteristicas(X_x2agregado)
# print(X_norm)
# print(mu)
# print(sigma)
plt.plot(X_norm[:, 0] + X_norm[:, 1] + X_norm[:, 2] + X_norm[:, 3] + X_norm[:, 4] + X_norm[:, 5] + X_norm[:, 6] +
         X_norm[:, 7], y, "k-", ms=1)
plt.xlabel('Cantidad de Votantes')
plt.ylabel('Cantidad de inscritos')
plt.show()

m = len(y)
X_x0agregado = np.column_stack((np.ones((m, 1)), X_norm)) # Añadir una columna de unos a x
# print(X_x0agregado)

print('Ejecutando descenso por el gradiente ...')
# Se establece un valor para alpha
alpha = 0.01
num_iteraciones = 16000

# inicializa theta y ejecuta descenso por el gradiente
# Lamentablemente es un arreglo muy complejo, no se puede resolver
theta = np.zeros((8, 1))
print(theta)
theta, J_history = rp.descensoGradienteMulti(X_x0agregado, y, theta, alpha, num_iteraciones)
print(theta)

plt.plot(range(J_history.size), J_history, "k-", linewidth = 2)
plt.xlabel('Numero de iteraciones')
plt.ylabel('Costo J')
plt.show(block=False)

# Mostrar los resultados del descenso por el gradiente
print('Theta calculado a partir del descenso por el gradiente: ')
print("{:f}, {:f}, {:f}".format(theta[0, 0], theta[1, 0], theta[2, 0], theta[3, 0], theta[4, 0], theta[5, 0],
                                theta[6, 0], theta[7, 0], theta[8, 0]))
print("")


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
