#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

double** allocate_matrix(int n) {
    double** mat = malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++)
        mat[i] = calloc(n, sizeof(double));
    return mat;
}

void free_matrix(double** mat, int n) {
    for (int i = 0; i < n; i++)
        free(mat[i]);
    free(mat);
}

void lu_decomposition(double** A, double** L, double** U, int N) {
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

void forward_substitution(double** L, double* b, double* y, int N) {
    for (int i = 0; i < N; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++)
            y[i] -= L[i][j] * y[j];
    }
}

void backward_substitution(double** U, double* y, double* x, int N) {
    for (int i = N - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < N; j++)
            x[i] -= U[i][j] * x[j];
        x[i] /= U[i][i];
    }
}

void invert_matrix(double** A, double** A_inv, int N) {
    double** L = allocate_matrix(N);
    double** U = allocate_matrix(N);
    lu_decomposition(A, L, U, N);
    for (int i = 0; i < N; i++) {
        double* e = calloc(N, sizeof(double));
        double* y = malloc(N * sizeof(double));
        double* x = malloc(N * sizeof(double));
        e[i] = 1.0;
        forward_substitution(L, e, y, N);
        backward_substitution(U, y, x, N);
        for (int j = 0; j < N; j++)
            A_inv[j][i] = x[j];
        free(e);
        free(y);
        free(x);
    }
    free_matrix(L, N);
    free_matrix(U, N);
}

void print_matrix(double** M, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%8.4f ", M[i][j]);
        printf("\n");
    }
}

int contar_columnas(const char* linea) {
    int count = 1;
    for (const char* p = linea; *p; p++)
        if (*p == ',') count++;
    return count;
}

double** leer_csv(const char* filename, int* N_out) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        perror("Error abriendo el archivo");
        return NULL;
    }

    char* line = malloc(32768);
    int filas = 0, N = 0;

    // Leer primera línea
    if (!fgets(line, 32768, fp)) {
        fprintf(stderr, "Archivo vacío\n");
        fclose(fp);
        free(line);
        return NULL;
    }

    N = contar_columnas(line);
    double** A = allocate_matrix(N);

    // Procesar primera fila
    int col = 0;
    char* token = strtok(line, ",\n");
    while (token && col < N)
        A[0][col++] = atof(token), token = strtok(NULL, ",\n");

    if (col != N) {
        fprintf(stderr, "Error: columnas incorrectas en fila 0\n");
        fclose(fp);
        free_matrix(A, N);
        free(line);
        return NULL;
    }

    filas = 1;

    // Leer el resto
    while (fgets(line, 32768, fp) && filas < N) {
        col = 0;
        token = strtok(line, ",\n");
        while (token && col < N)
            A[filas][col++] = atof(token), token = strtok(NULL, ",\n");
        if (col != N) {
            fprintf(stderr, "Error: columnas incorrectas en fila %d\n", filas);
            fclose(fp);
            free_matrix(A, N);
            free(line);
            return NULL;
        }
        filas++;
    }

    fclose(fp);
    free(line);

    if (filas != N) {
        fprintf(stderr, "Error: filas insuficientes\n");
        free_matrix(A, N);
        return NULL;
    }

    *N_out = N;
    return A;
}

int main() {
    int N;
    double** A = leer_csv("matriz.csv", &N);
    if (!A) return 1;

    double** A_inv = allocate_matrix(N);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    invert_matrix(A, A_inv, N);

    gettimeofday(&end, NULL);
    double t = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    printf("Matriz inversa A^-1:\n");
    print_matrix(A_inv, N);
    printf("\nTiempo de ejecución: %.6f segundos\n", t);

    free_matrix(A, N);
    free_matrix(A_inv, N);
    return 0;
}

