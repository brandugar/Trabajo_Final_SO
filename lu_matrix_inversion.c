#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 3 

// Paso 1: Descomposición LU (sin pivoteo)
void lu_decomposition(double A[N][N], double L[N][N], double U[N][N]) {
    for (int i = 0; i < N; i++) {
        // U
        for (int k = i; k < N; k++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++)
                sum += L[i][j] * U[j][k];
            U[i][k] = A[i][k] - sum;
        }

        // L
        for (int k = i; k < N; k++) {
            if (i == k)
                L[i][i] = 1.0;
            else {
                double sum = 0.0;
                for (int j = 0; j < i; j++)
                    sum += L[k][j] * U[j][i];
                L[k][i] = (A[k][i] - sum) / U[i][i];
            }
        }
    }
}

// Paso 2: Sustitución hacia adelante (Ly = b)
void forward_substitution(double L[N][N], double b[N], double y[N]) {
    for (int i = 0; i < N; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++)
            y[i] -= L[i][j] * y[j];
        // L[i][i] == 1
    }
}

// Paso 3: Sustitución hacia atrás (Ux = y)
void backward_substitution(double U[N][N], double y[N], double x[N]) {
    for (int i = N - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < N; j++)
            x[i] -= U[i][j] * x[j];
        x[i] /= U[i][i];
    }
}

// Paso 4: Inversión usando LU
void invert_matrix(double A[N][N], double A_inv[N][N]) {
    double L[N][N] = {0}, U[N][N] = {0};
    lu_decomposition(A, L, U);

    for (int i = 0; i < N; i++) {
        double e[N] = {0}, y[N], x[N];
        e[i] = 1.0;

        forward_substitution(L, e, y);
        backward_substitution(U, y, x);

        for (int j = 0; j < N; j++)
            A_inv[j][i] = x[j];  // Columna i
    }
}

// Utilidad: imprimir una matriz
void print_matrix(double M[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%8.4f ", M[i][j]);
        printf("\n");
    }
}

int main() {
    double A[N][N] = {
        {4, 3, 2},
        {2, 1, 1},
        {6, 7, 5}
    };

    double A_inv[N][N];
    invert_matrix(A, A_inv);

    printf("Matriz inversa A^-1:\n");
    print_matrix(A_inv);

    return 0;
}
