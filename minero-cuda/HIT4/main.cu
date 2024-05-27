#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Incluye la declaración de la función md5Hash
#include "md5.cu"

__global__ void md5Kernel(unsigned char* data, int length, uint32_t *a1, uint32_t *b1, uint32_t *c1, uint32_t *d1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        // Calcula el hash MD5 utilizando la función definida en md5.cu
        md5Hash(data, length, a1, b1, c1, d1);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Uso: %s <cadena>\n", argv[0]);
        return 1;
    }

    // Obtiene la cadena del argumento de línea de comandos
    char* input_string = argv[1];
    int string_length = strlen(input_string);

    // Copia la cadena a la GPU
    unsigned char* d_input_string;
    cudaMalloc((void**)&d_input_string, string_length);
    cudaMemcpy(d_input_string, input_string, string_length, cudaMemcpyHostToDevice);

    // Variables para almacenar el resultado del hash MD5
    uint32_t* d_a, * d_b, * d_c, * d_d;
    cudaMalloc((void**)&d_a, sizeof(uint32_t));
    cudaMalloc((void**)&d_b, sizeof(uint32_t));
    cudaMalloc((void**)&d_c, sizeof(uint32_t));
    cudaMalloc((void**)&d_d, sizeof(uint32_t));

    // Llama al kernel para calcular el hash MD5
    md5Kernel<<<(string_length + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_input_string, string_length, d_a, d_b, d_c, d_d);

    // Copia los resultados de vuelta desde la GPU
    uint32_t a, b, c, d;
    cudaMemcpy(&a, d_a, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b, d_b, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c, d_c, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&d, d_d, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Imprime el hash MD5 calculado
    printf("Hash MD5 de '%s' es: %08x%08x%08x%08x\n", input_string, a, b, c, d);

    // Libera la memoria de la GPU
    cudaFree(d_input_string);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);

    return 0;
}
