import numpy as np
import csv
from numba import njit, prange
import time


@njit(parallel=True)
def gauss_jordan_inverse_numba(A):
    n = len(A)
    A = A.astype(np.float64)

    I = np.identity(n, dtype=np.float64)
    AI = np.hstack((A, I))  # Matriz aumentada [A | I]

    for i in range(n):
        # Pivoteo parcial (reemplazo del swap)
        max_row = i + np.argmax(np.abs(AI[i:, i]))
        temp = AI[i].copy()
        AI[i] = AI[max_row]
        AI[max_row] = temp

        # Normalizar la fila pivote
        AI[i] = AI[i] / AI[i, i]

        # Eliminar los elementos arriba y abajo del pivote (paralelo)
        for j in prange(n):
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
inv_A = gauss_jordan_inverse_numba(A)
fin = time.time()
print(f"Tiempo de ejecución: {fin - inicio:.6f} segundos")
# print("Inversa por Gauss-Jordan:")
# print(inv_A)
