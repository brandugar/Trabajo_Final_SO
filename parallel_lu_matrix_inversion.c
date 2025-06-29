#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <string.h>

#define MAX_N 1024
#define MAX_LINE 8192

typedef struct {
    int col;
    int N;
    double **L;
    double **U;
    double **A_inv;
} ThreadData;

double **alocar_matriz(int N) {
    double **M = malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++)
        M[i] = malloc(N * sizeof(double));
    return M;
}

void liberar_matriz(double **M, int N) {
    for (int i = 0; i < N; i++)
        free(M[i]);
    free(M);
}

int leer_csv(const char *filename, double ***A_out, int *N_out) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error al abrir archivo CSV");
        return 0;
    }

    char line[MAX_LINE];
    int N = 0;
    int columnas = -1;
    double **A = alocar_matriz(MAX_N);

    while (fgets(line, sizeof(line), fp)) {
        int col = 0;
        char *token = strtok(line, ",\n");
        while (token) {
            A[N][col++] = atof(token);
            token = strtok(NULL, ",\n");
        }
        if (columnas == -1) columnas = col;
        else if (col != columnas) {
            fprintf(stderr, "Error: filas con distinta cantidad de columnas.\n");
            liberar_matriz(A, MAX_N);
            fclose(fp);
            return 0;
        }
        N++;
    }

    fclose(fp);
    if (N != columnas) {
        fprintf(stderr, "Error: matriz no es cuadrada.\n");
        liberar_matriz(A, MAX_N);
        return 0;
    }

    *A_out = A;
    *N_out = N;
    return 1;
}

void lu_decomposition(double **A, double **L, double **U, int N) {
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

void forward_substitution(double **L, double *b, double *y, int N) {
    for (int i = 0; i < N; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++)
            y[i] -= L[i][j] * y[j];
    }
}

void backward_substitution(double **U, double *y, double *x, int N) {
    for (int i = N - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < N; j++)
            x[i] -= U[i][j] * x[j];
        x[i] /= U[i][i];
    }
}

void *invert_column(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    int i = data->col, N = data->N;
    double *e = calloc(N, sizeof(double));
    double *y = malloc(N * sizeof(double));
    double *x = malloc(N * sizeof(double));

    e[i] = 1.0;
    forward_substitution(data->L, e, y, N);
    backward_substitution(data->U, y, x, N);

    for (int j = 0; j < N; j++)
        data->A_inv[j][i] = x[j];

    free(e); free(y); free(x);
    return NULL;
}

void invertir_matriz(double **A, double **A_inv, int N) {
    double **L = alocar_matriz(N);
    double **U = alocar_matriz(N);
    lu_decomposition(A, L, U, N);

    pthread_t *threads = malloc(N * sizeof(pthread_t));
    ThreadData *data = malloc(N * sizeof(ThreadData));

    for (int i = 0; i < N; i++) {
        data[i].col = i;
        data[i].N = N;
        data[i].L = L;
        data[i].U = U;
        data[i].A_inv = A_inv;
        pthread_create(&threads[i], NULL, invert_column, &data[i]);
    }

    for (int i = 0; i < N; i++)
        pthread_join(threads[i], NULL);

    liberar_matriz(L, N);
    liberar_matriz(U, N);
    free(threads);
    free(data);
}

void imprimir_matriz(double **M, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%10.6f ", M[i][j]);
        printf("\n");
    }
}

int main() {
    double **A, **A_inv;
    int N;

    if (!leer_csv("matriz.csv", &A, &N)) {
        fprintf(stderr, "Error leyendo matriz desde CSV.\n");
        return 1;
    }

    A_inv = alocar_matriz(N);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    invertir_matriz(A, A_inv, N);

    gettimeofday(&end, NULL);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    printf("Matriz inversa A^-1:\n");
    imprimir_matriz(A_inv, N);

    printf("\nTiempo de ejecución: %.6f segundos\n", elapsed);

    liberar_matriz(A, N);
    liberar_matriz(A_inv, N);

    return 0;
}
