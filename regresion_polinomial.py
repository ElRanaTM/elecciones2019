# Regresion lineal con multiples variables

import numpy as np


def calcularCosto(X, y, theta):
    J = 0
    s = np.power((X.dot(theta) - np.transpose([y])), 2)
    #print("INICIO DENTRO COSTO")
    #print(s)
    #print(s.sum(axis = 0))
    #print("FIN DENTRO COSTO")
    J = (1.0 / (2 * m)) * s.sum(axis = 0)

    return J


def descensoGradienteMulti(X, y, theta, alpha, num_iteraciones):
    m = len(y) # numero de ejemplos de entrenamiento
    J_history = np.zeros((num_iteraciones, 1))

    for i in range(num_iteraciones):
        theta = theta - alpha * (1.0 / m) * np.transpose(X).dot(X.dot(theta) - np.transpose([y]))
        # Guardar el costo de J en cada iteracion
        J_history[i] = calcularCosto(X, y, theta)
        # print(J_history[i])

    return theta, J_history


def normalizarCaracteristicas(X):
    # Se requiere establecer los valores de manera correcta
    X_norm = X
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))

    for i in range(X.shape[1]):
        mu[:, i] = np.mean(X[:, i])
        sigma[:, i] = np.std(X[:, i])
        X_norm[:, i] = (X[:, i] - float(mu[:, i])) / float(sigma[:, i])

    return X_norm, mu, sigma
