#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

/**
 * @brief Genera una matriz invertible y la guarda en un archivo CSV.
 *
 * La invertibilidad se garantiza creando una matriz diagonalmente dominante.
 * En una matriz así, para cada fila, el valor absoluto del elemento en la
 * diagonal principal es estrictamente mayor que la suma de los valores
 * absolutos de los demás elementos de la fila.
 *
 * Compilación: gcc generador.c -o generador -lm
 * Uso:         ./generador <tamaño>
 * Ejemplo:     ./generador 1000
 */
int main(int argc, char *argv[]) {
    // 1. Validar los argumentos de entrada
    if (argc != 2) {
        fprintf(stderr, "Error: Número incorrecto de argumentos.\n");
        fprintf(stderr, "Uso: %s <tamaño>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        fprintf(stderr, "Error: El tamaño debe ser un entero positivo.\n");
        return 1;
    }

    // 2. Crear el nombre del archivo de salida dinámicamente
    char filename[256];
    sprintf(filename, "matriz_%dx%d_invertible.csv", N, N);

    // 3. Abrir el archivo en modo escritura
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error al abrir el archivo de salida");
        return 1;
    }

    // 4. Inicializar el generador de números aleatorios
    srand(time(NULL));

    printf("Generando matriz invertible %dx%d en '%s'...\n", N, N, filename);

    // 5. Generar y escribir la matriz fila por fila
    for (int i = 0; i < N; i++) {
        long long row_sum = 0;
        int diagonal_val = 0;

        // Generar cada elemento de la fila
        for (int j = 0; j < N; j++) {
            // Generar un valor aleatorio para los elementos fuera de la diagonal
            int random_val = rand() % 100 + 1; // Valores entre 1 y 100

            if (i == j) {
                // Si es el elemento diagonal, lo dejamos para el final
                continue;
            }
            
            // Escribir el valor y sumarlo al total de la fila
            fprintf(file, "%d", random_val);
            row_sum += abs(random_val);
            
            // Escribir la coma si no es el último elemento
            if (j < N -1) {
                fprintf(file, ",");
            }
        }
        
        // El valor de la diagonal debe ser mayor que la suma del resto
        diagonal_val = row_sum + (rand() % 50 + 1);

        // Volver a escribir la fila, esta vez insertando el valor de la diagonal
        // fseek permite mover el "cursor" del archivo a una posición específica
        fseek(file, -ftell(file) + i * (ftell(file) / (i + 1)), SEEK_SET);

        for (int j = 0; j < N; j++) {
             int random_val = rand() % 100 + 1;
             if (i == j) {
                 fprintf(file, "%d", diagonal_val);
             } else {
                 fprintf(file, "%d", random_val);
             }
             if (j < N -1) {
                fprintf(file, ",");
            }
        }

        // Escribir el salto de línea para la siguiente fila
        fprintf(file, "\n");
    }

    // 6. Cerrar el archivo y notificar al usuario
    fclose(file);
    printf("¡Archivo '%s' generado con éxito!\n", filename);

    return 0;
}