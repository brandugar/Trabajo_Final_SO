import numpy as np
import time
import csv


def gauss_jordan_inverse(A):
    n = len(A)
    A = A.astype(float)

    I = np.identity(n)
    AI = np.hstack((A, I))  # Matriz aumentada [A | I]

    for i in range(n):
        # Pivoteo parcial
        max_row = np.argmax(np.abs(AI[i:, i])) + i
        AI[[i, max_row]] = AI[[max_row, i]]

        # Normalizar la fila pivote
        AI[i] = AI[i] / AI[i, i]

        # Eliminar los elementos arriba y abajo del pivote
        for j in range(n):
            if i != j:
                AI[j] = AI[j] - AI[j, i] * AI[i]

    return AI[:, n:]  # Retornar la parte derecha: la inversa


def leer_matriz_desde_csv(nombre_archivo):
    with open(nombre_archivo, 'r') as archivo:
        lector = csv.reader(archivo)
        matriz = [list(map(float, fila)) for fila in lector]
    return np.array(matriz)


# Uso del programa
nombre_archivo = '../../matrices/matriz_5000x5000_invertible.csv'
A = leer_matriz_desde_csv(nombre_archivo)
# print("Matriz original:")
# print(A)

inicio = time.time()
inv_A = gauss_jordan_inverse(A)
fin = time.time()
print(f"Tiempo de ejecución: {fin - inicio:.6f} segundos")
# print("Inversa por Gauss-Jordan:")
# print(inv_A)
