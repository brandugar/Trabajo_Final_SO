#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define N 3
#define NUM_THREADS N

typedef struct {
    int col;
    double (*L)[N];
    double (*U)[N];
    double (*A_inv)[N];
} ThreadData;

double A[N][N] = {
    {4, 3, 2},
    {2, 1, 1},
    {6, 7, 5}
};

void lu_decomposition(double A[N][N], double L[N][N], double U[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int k = i; k < N; k++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++)
                sum += L[i][j] * U[j][k];
            U[i][k] = A[i][k] - sum;
        }

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

void forward_substitution(double L[N][N], double b[N], double y[N]) {
    for (int i = 0; i < N; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++)
            y[i] -= L[i][j] * y[j];
    }
}

void backward_substitution(double U[N][N], double y[N], double x[N]) {
    for (int i = N - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < N; j++)
            x[i] -= U[i][j] * x[j];
        x[i] /= U[i][i];
    }
}

// Hilo para calcular una columna de la inversa
void *invert_column(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    int i = data->col;
    double *y = malloc(sizeof(double) * N);
    double *x = malloc(sizeof(double) * N);
    double e[N] = {0};
    e[i] = 1.0;

    forward_substitution(data->L, e, y);
    backward_substitution(data->U, y, x);

    for (int j = 0; j < N; j++)
        data->A_inv[j][i] = x[j];

    free(y);
    free(x);
    return NULL;
}

// Matriz inversa con hilos
void invert_matrix_parallel(double A[N][N], double A_inv[N][N]) {
    double L[N][N] = {0}, U[N][N] = {0};
    lu_decomposition(A, L, U);

    pthread_t threads[NUM_THREADS];
    ThreadData data[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        data[i].col = i;
        data[i].L = L;
        data[i].U = U;
        data[i].A_inv = A_inv;
        pthread_create(&threads[i], NULL, invert_column, &data[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
}

// Utilidad para imprimir matrices
void print_matrix(double M[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%8.4f ", M[i][j]);
        printf("\n");
    }
}

int main() {
    double A_inv[N][N];
    invert_matrix_parallel(A, A_inv);

    printf("Matriz inversa A^-1 (paralela):\n");
    print_matrix(A_inv);

    return 0;
}
