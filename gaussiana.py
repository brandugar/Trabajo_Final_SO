import numpy as np


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


# Ejemplo de uso
A = np.array([[2, 1], [5, 3]], dtype=float)
inv_A = gauss_jordan_inverse(A)
print("Inversa por Gauss-Jordan:")
print(inv_A)
