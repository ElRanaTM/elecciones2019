# Archivo para las impresiones de los graficos
import matplotlib.pyplot as plt


def imprimirCC(x, y):
    plt.plot(x[:, 0], y, "y.", ms=1)
    plt.title("Comunidad Ciudadana")
    plt.xlabel('Cantidad de Votantes')
    plt.ylabel('Cantidad de inscritos')
    plt.show()


def imprimirFPV(x, y):
    plt.plot(x[:, 1], y, "m.", ms=1)
    plt.title("Frente Para la Victoria")
    plt.xlabel('Cantidad de Votantes')
    plt.ylabel('Cantidad de inscritos')
    plt.show()


def imprimirMTS(x, y):
    plt.plot(x[:, 2], y, "g.", ms=1)
    plt.title("Movimiento Tercer Sistema")
    plt.xlabel('Cantidad de Votantes')
    plt.ylabel('Cantidad de inscritos')
    plt.show()


def imprimirMAS(x, y):
    plt.plot(x[:, 3], y, "b.", ms=1)
    plt.title("Movimiento Al Socialismo")
    plt.xlabel('Cantidad de Votantes')
    plt.ylabel('Cantidad de inscritos')
    plt.show()


def imprimir21F(x, y):
    plt.plot(x[:, 4], y, "r.", ms=1)
    plt.title("Bolivia Dice No 21F")
    plt.xlabel('Cantidad de Votantes')
    plt.ylabel('Cantidad de inscritos')
    plt.show()


def imprimirPDC(x, y):
    plt.plot(x[:, 5], y, "k.", ms=1)
    plt.title("Partido Democrata Cristiano")
    plt.xlabel('Cantidad de Votantes')
    plt.ylabel('Cantidad de inscritos')
    plt.show()


def imprimirMNR(x, y):
    plt.plot(x[:, 6], y, "c.", ms=1)
    plt.title("Movimiento Nacionalista Revolucionario")
    plt.xlabel('Cantidad de Votantes')
    plt.ylabel('Cantidad de inscritos')
    plt.show()


def imprimirPAN(x, y):
    plt.plot(x[:, 7], y, "r.", ms=1)
    plt.title("Partido de Accion Nacional Boliviano")
    plt.xlabel('Cantidad de Votantes')
    plt.ylabel('Cantidad de inscritos')
    plt.show()




'''
plt.plot(X_x2agregado[:, 7], y, "r.", ms=1)
plt.title("Partido de Accion Nacional Boliviano")
plt.xlabel('Cantidad de Votantes')
plt.ylabel('Cantidad de inscritos')
plt.show()
'''