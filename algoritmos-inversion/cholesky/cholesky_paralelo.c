#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define MAX_SIZE 500
#define RUTA_BASE "../../matrices/def_pos_inv/"

// Leer CSV
int leer_csv(const char *ruta_archivo, double A[MAX_SIZE][MAX_SIZE], int *n) {
    FILE *f = fopen(ruta_archivo, "r");
    if (!f) {
        perror("No se pudo abrir el archivo CSV");
        return 0;
    }

    char linea[8192];
    int fila = 0;
    while (fgets(linea, sizeof(linea), f)) {
        int col = 0;
        char *token = strtok(linea, ",");
        while (token && col < MAX_SIZE) {
            A[fila][col++] = atof(token);
            token = strtok(NULL, ",");
        }
        fila++;
        if (fila > MAX_SIZE) break;
    }

    *n = fila;
    fclose(f);
    return 1;
}

// Imprimir matriz
void imprimir(double A[MAX_SIZE][MAX_SIZE], int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%10.4lf ", A[i][j]);
        }
        printf("\n");
    }
}

// Descomposición de Cholesky
int cholesky(double A[MAX_SIZE][MAX_SIZE], double L[MAX_SIZE][MAX_SIZE], int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double suma = 0;
            for (int k = 0; k < j; k++)
                suma += L[i][k] * L[j][k];

            if (i == j) {
                double val = A[i][i] - suma;
                if (val <= 0) return 0;
                L[i][j] = sqrt(val);
            } else {
                L[i][j] = (1.0 / L[j][j]) * (A[i][j] - suma);
            }
        }
    }
    return 1;
}

// Ly = b
void forward_sub(double L[MAX_SIZE][MAX_SIZE], double b[MAX_SIZE], double y[MAX_SIZE], int n) {
    for (int i = 0; i < n; i++) {
        double suma = 0;
        for (int j = 0; j < i; j++)
            suma += L[i][j] * y[j];
        y[i] = (b[i] - suma) / L[i][i];
    }
}

// Lᵗx = y
void backward_sub(double L[MAX_SIZE][MAX_SIZE], double y[MAX_SIZE], double x[MAX_SIZE], int n) {
    for (int i = n - 1; i >= 0; i--) {
        double suma = 0;
        for (int j = i + 1; j < n; j++)
            suma += L[j][i] * x[j];
        x[i] = (y[i] - suma) / L[i][i];
    }
}

// Inversión con paralelismo en el ciclo por columnas
int invertir_cholesky(double A[MAX_SIZE][MAX_SIZE], double A_inv[MAX_SIZE][MAX_SIZE], int n) {
    double L[MAX_SIZE][MAX_SIZE] = {0};
    if (!cholesky(A, L, n)) return 0;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double e[MAX_SIZE] = {0}, y[MAX_SIZE], x[MAX_SIZE];
        e[i] = 1.0;

        forward_sub(L, e, y, n);
        backward_sub(L, y, x, n);

        for (int j = 0; j < n; j++)
            A_inv[j][i] = x[j];
    }

    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Uso: %s <archivo_csv>\n", argv[0]);
        return 1;
    }

    char ruta_completa[1024];
    snprintf(ruta_completa, sizeof(ruta_completa), "%s%s", RUTA_BASE, argv[1]);

    double A[MAX_SIZE][MAX_SIZE], A_inv[MAX_SIZE][MAX_SIZE];
    int n;

    if (!leer_csv(ruta_completa, A, &n)) {
        return 1;
    }

    double inicio = omp_get_wtime();

    if (!invertir_cholesky(A, A_inv, n)) {
        printf("La matriz no es definida positiva o no es invertible.\n");
        return 1;
    }

    double fin = omp_get_wtime();
    double tiempo = fin - inicio;

    printf("Tiempo de inversión: %.6f segundos\n", tiempo);

    if (n <= 10) {
        printf("Matriz inversa:\n");
        imprimir(A_inv, n);
    } else {
        printf("Matriz mayor a 10x10. Inversa no mostrada en pantalla.\n");
    }

    return 0;
}

