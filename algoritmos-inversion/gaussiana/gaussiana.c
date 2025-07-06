#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define MAX_LINE_LEN 250000 // Búfer para leer una línea del CSV (ajustar si las filas son muy largas)

/**
 * @brief Determina la dimensión N de una matriz NxN desde un archivo CSV.
 * Cuenta el número de valores en la primera línea.
 * @param filename Nombre del archivo CSV.
 * @return La dimensión N de la matriz, o -1 si hay un error.
 */
int get_matrix_size(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error al abrir el archivo");
        return -1;
    }

    char line[MAX_LINE_LEN];
    if (fgets(line, sizeof(line), file) == NULL) {
        fclose(file);
        return 0; // Archivo vacío
    }
    fclose(file);

    int count = 0;
    char* token = strtok(line, ",");
    while (token) {
        count++;
        token = strtok(NULL, ",");
    }
    return count;
}


/**
 * @brief Lee una matriz desde un archivo CSV a la mitad izquierda de una matriz ya alocada.
 * @param filename Nombre del archivo.
 * @param matrix Puntero a la matriz (debe tener 'n' filas y al menos 'n' columnas).
 * @param n Dimensión de la matriz.
 */
void read_matrix_from_csv(const char* filename, double** matrix, int n) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error al abrir el archivo para lectura");
        exit(EXIT_FAILURE);
    }

    char line[MAX_LINE_LEN];
    int row = 0;
    while (fgets(line, sizeof(line), file) && row < n) {
        char* token = strtok(line, ",");
        int col = 0;
        while (token && col < n) {
            matrix[row][col] = atof(token); // atof convierte string a double
            token = strtok(NULL, ",");
            col++;
        }
        row++;
    }
    fclose(file);
}

/**
 * @brief Aplica el método de Gauss-Jordan para invertir una matriz.
 * Opera sobre una matriz aumentada [A|I] y la transforma en [I|A^-1].
 * @param AI La matriz aumentada de tamaño n x 2n.
 * @param n La dimensión de la matriz original A.
 * @return 0 si tiene éxito, -1 si la matriz es singular.
 */
int gauss_jordan_inverse(double** AI, int n) {
    for (int i = 0; i < n; i++) {
        // 1. Pivoteo parcial: encontrar la fila con el máximo valor en la columna i
        int max_row = i;
        for (int k = i + 1; k < n; k++) {
            if (fabs(AI[k][i]) > fabs(AI[max_row][i])) {
                max_row = k;
            }
        }

        // Intercambiar la fila actual con la fila del pivote máximo
        double* temp = AI[i];
        AI[i] = AI[max_row];
        AI[max_row] = temp;

        // Verificar si la matriz es singular
        if (fabs(AI[i][i]) < 1e-10) {
            return -1; // Matriz singular, no se puede invertir
        }

        // 2. Normalizar la fila pivote (dividir toda la fila por el elemento pivote)
        double pivot = AI[i][i];
        for (int j = i; j < 2 * n; j++) {
            AI[i][j] /= pivot;
        }

        // 3. Eliminar los otros elementos en la columna pivote
        for (int k = 0; k < n; k++) {
            if (i != k) {
                double factor = AI[k][i];
                for (int j = i; j < 2 * n; j++) {
                    AI[k][j] -= factor * AI[i][j];
                }
            }
        }
    }
    return 0; // Éxito
}

int main() {
    // Ajusta la ruta al archivo CSV según sea necesario
    const char* filename = "../../matrices/matriz_5000x5000_invertible.csv";

    printf("Determinando el tamaño de la matriz desde %s...\n", filename);
    int n = get_matrix_size(filename);
    if (n <= 0) {
        fprintf(stderr, "No se pudo determinar un tamaño de matriz válido.\n");
        return 1;
    }
    printf("Matriz detectada de tamaño %d x %d.\n", n, n);

    // --- Asignación de Memoria ---
    printf("Asignando memoria para la matriz aumentada [%d x %d]...\n", n, 2 * n);
    double** AI = (double**)malloc(n * sizeof(double*));
    if (AI == NULL) {
        perror("Fallo en malloc para las filas");
        return 1;
    }
    for (int i = 0; i < n; i++) {
        AI[i] = (double*)malloc(2 * n * sizeof(double));
        if (AI[i] == NULL) {
            perror("Fallo en malloc para las columnas");
            // Liberar memoria ya asignada
            for (int k = 0; k < i; k++) free(AI[k]);
            free(AI);
            return 1;
        }
    }

    // --- Inicialización de la Matriz Aumentada [A|I] ---
    printf("Leyendo matriz desde el archivo CSV...\n");
    read_matrix_from_csv(filename, AI, n);

    printf("Inicializando la matriz identidad...\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            AI[i][j + n] = (i == j) ? 1.0 : 0.0;
        }
    }

    // --- Inversión y Medición de Tiempo ---
    printf("Iniciando inversión por Gauss-Jordan...\n");
    clock_t start = clock();
    int result = gauss_jordan_inverse(AI, n);
    clock_t end = clock();

    if (result != 0) {
        fprintf(stderr, "Error: La matriz es singular y no puede ser invertida.\n");
    } else {
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("---------------------------------------------------\n");
        printf("Tiempo de ejecución: %f segundos\n", time_spent);
        printf("---------------------------------------------------\n");
    }
    
    // --- Liberación de Memoria ---
    printf("Liberando memoria...\n");
    for (int i = 0; i < n; i++) {
        free(AI[i]);
    }
    free(AI);

    printf("Programa finalizado.\n");
    return 0;
}
