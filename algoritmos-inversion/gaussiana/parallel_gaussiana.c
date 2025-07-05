#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h> // Librería para hilos

// Define el número de hilos que se usarán.
// Un buen punto de partida es el número de núcleos de tu CPU.
#define NUM_THREADS 4

#define MAX_LINE_LEN 65536

// --- Estructura para pasar argumentos a los hilos ---
typedef struct {
    long id;        // Identificador del hilo (0 a NUM_THREADS-1)
    int n;          // Dimensión de la matriz
    double** AI;    // Puntero a la matriz aumentada
} thread_args_t;

// --- Variables Globales para la Sincronización ---
pthread_barrier_t barrier; // Barrera para sincronizar hilos en cada iteración

// (Las funciones get_matrix_size y read_matrix_from_csv permanecen iguales)
int get_matrix_size(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error al abrir el archivo");
        return -1;
    }
    char line[MAX_LINE_LEN];
    if (fgets(line, sizeof(line), file) == NULL) {
        fclose(file);
        return 0;
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
            matrix[row][col] = atof(token);
            token = strtok(NULL, ",");
            col++;
        }
        row++;
    }
    fclose(file);
}


/**
 * @brief Función ejecutada por cada hilo trabajador.
 * Realiza el paso de eliminación en un subconjunto de filas.
 */
void* elimination_worker(void* arg) {
    thread_args_t* args = (thread_args_t*)arg;
    long id = args->id;
    int n = args->n;
    double** AI = args->AI;

    for (int i = 0; i < n; i++) { // Bucle sobre las filas pivote
        // Sincronización: Esperar a que el hilo principal haga el pivoteo y la normalización.
        pthread_barrier_wait(&barrier);

        // Calcular el rango de filas para este hilo
        int start_row = id * (n / NUM_THREADS);
        int end_row = (id + 1) * (n / NUM_THREADS);
        if (id == NUM_THREADS - 1) {
            end_row = n; // El último hilo toma las filas restantes
        }
        
        // 3. Eliminar los otros elementos en la columna pivote (para las filas asignadas)
        for (int k = start_row; k < end_row; k++) {
            if (i != k) {
                double factor = AI[k][i];
                // Optimización: si el factor ya es cero, no hay nada que hacer
                if (fabs(factor) > 1e-12) {
                    for (int j = i; j < 2 * n; j++) {
                        AI[k][j] -= factor * AI[i][j];
                    }
                }
            }
        }
        
        // Sincronización: Esperar a que todos los hilos terminen la eliminación para la pivote 'i'.
        pthread_barrier_wait(&barrier);
    }
    return NULL;
}

/**
 * @brief Aplica el método de Gauss-Jordan de forma paralela.
 */
int gauss_jordan_parallel(double** AI, int n) {
    pthread_t threads[NUM_THREADS];
    thread_args_t args[NUM_THREADS];

    // LA CORRECCIÓN ESTÁ AQUÍ:
    // Inicializar la barrera para NUM_THREADS (hilos trabajadores) + 1 (hilo principal).
    pthread_barrier_init(&barrier, NULL, NUM_THREADS + 1);

    // Crear los hilos trabajadores
    for (long i = 0; i < NUM_THREADS; i++) {
        args[i].id = i;
        args[i].n = n;
        args[i].AI = AI;
        pthread_create(&threads[i], NULL, elimination_worker, &args[i]);
    }

    // El hilo principal orquesta y realiza los pasos secuenciales
    for (int i = 0; i < n; i++) {
        // ... (el resto de la función es correcto) ...
        
        // Sincronización: Liberar a los hilos trabajadores
        pthread_barrier_wait(&barrier);

        // Sincronización: Esperar a que la eliminación termine
        pthread_barrier_wait(&barrier);
    }

    // Unir los hilos trabajadores
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    // Destruir la barrera
    pthread_barrier_destroy(&barrier);

    return 0; // Éxito
}

int main() {
    const char* filename = "../../matrices/matriz_5000x5000_invertible.csv";
    
    printf("Usando %d hilos de trabajo.\n", NUM_THREADS);
    printf("Determinando el tamaño de la matriz desde %s...\n", filename);
    int n = get_matrix_size(filename);
    if (n <= 0) {
        fprintf(stderr, "No se pudo determinar un tamaño de matriz válido.\n");
        return 1;
    }
    printf("Matriz detectada de tamaño %d x %d.\n", n, n);

    printf("Asignando memoria para la matriz aumentada [%d x %d]...\n", n, 2 * n);
    double** AI = (double**)malloc(n * sizeof(double*));
    if (!AI) { perror("malloc"); return 1; }
    for (int i = 0; i < n; i++) {
        AI[i] = (double*)malloc(2 * n * sizeof(double));
        if (!AI[i]) { perror("malloc"); return 1; }
    }

    printf("Leyendo matriz desde el archivo CSV...\n");
    read_matrix_from_csv(filename, AI, n);

    printf("Inicializando la matriz identidad...\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            AI[i][j + n] = (i == j) ? 1.0 : 0.0;
        }
    }

    printf("Iniciando inversión por Gauss-Jordan (Paralelo)...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int result = gauss_jordan_parallel(AI, n);
    
    clock_gettime(CLOCK_MONOTONIC, &end);

    if (result != 0) {
        fprintf(stderr, "Error: La matriz es singular y no puede ser invertida.\n");
    } else {
        double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("---------------------------------------------------\n");
        printf("Tiempo de ejecución: %f segundos\n", time_spent);
        printf("---------------------------------------------------\n");
    }
    
    printf("Liberando memoria...\n");
    for (int i = 0; i < n; i++) {
        free(AI[i]);
    }
    free(AI);

    printf("Programa finalizado.\n");
    return 0;
}
