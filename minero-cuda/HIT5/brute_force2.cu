#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda.h>

// Incluir el archivo md5.cu
#include "md5.cu"

__device__ void int_to_str(uint32_t num, char* str) {
    char buffer[12]; // Suficiente para almacenar el número más grande como una cadena
    char* p = buffer;
    do {
        *p++ = '0' + num % 10;
        num /= 10;
    } while (num != 0);

    // Invertir la cadena
    while (p != buffer) {
        *str++ = *--p;
    }
    *str = '\0';
}

// Función para convertir uint32_t a string hexadecimal
__device__ void uint32_to_hex_str(uint32_t num, char* str) {
    const char hex_chars[] = "0123456789abcdef";
    for (int i = 0; i < 8; ++i) {
        str[7 - i] = hex_chars[num & 0x0F];
        num >>= 4;
    }
    str[8] = '\0';
}

// Función para verificar si el hash empieza con el prefijo deseado
__device__ bool hash_starts_with(uint32_t a, uint32_t b, uint32_t c, uint32_t d, const char* prefix, int prefix_len) {
    char hash_str[33];
    uint32_to_hex_str(a, hash_str);
    uint32_to_hex_str(b, hash_str + 8);
    uint32_to_hex_str(c, hash_str + 16);
    uint32_to_hex_str(d, hash_str + 24);
    hash_str[32] = '\0';

    for (int i = 0; i < prefix_len; ++i) {
        if (hash_str[i] != prefix[i]) {
            return false;
        }
    }
    return true;
}

// Kernel de CUDA
__global__ void find_nonce(char* base_str, size_t base_len, const char* prefix, int prefix_len, uint32_t* result_nonce, uint32_t* result_a, uint32_t* result_b, uint32_t* result_c, uint32_t* result_d) {
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    char data[64];

    uint32_t a, b, c, d;
    for (uint32_t nonce = id + blockDim.x * gridDim.x * threadIdx.y + threadIdx.x; ; nonce += blockDim.x * gridDim.x * blockDim.y * gridDim.y) {
        int data_len = base_len;
        int_to_str(nonce, data + base_len);

        memcpy(data, base_str, base_len);

        md5Hash((unsigned char*)data, data_len, &a, &b, &c, &d);
        if (hash_starts_with(a, b, c, d, prefix, prefix_len)) {
            *result_nonce = nonce;
            *result_a = a;
            *result_b = b;
            *result_c = c;
            *result_d = d;
            break;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <target_prefix> <base_string>\n", argv[0]);
        return 1;
    }

    const char* target_prefix = argv[1];
    const char* base_string = argv[2];
    size_t base_len = strlen(base_string);
    int prefix_len = strlen(target_prefix);

    char* d_base_string;
    char* d_target_prefix;
    uint32_t* d_result_nonce;
    uint32_t* d_result_a;
    uint32_t* d_result_b;
    uint32_t* d_result_c;
    uint32_t* d_result_d;

    cudaMalloc(&d_base_string, base_len + 1);
    cudaMemcpy(d_base_string, base_string, base_len + 1, cudaMemcpyHostToDevice);

    cudaMalloc(&d_target_prefix, prefix_len + 1);
    cudaMemcpy(d_target_prefix, target_prefix, prefix_len + 1, cudaMemcpyHostToDevice);

    cudaMalloc(&d_result_nonce, sizeof(uint32_t));
    cudaMalloc(&d_result_a, sizeof(uint32_t));
    cudaMalloc(&d_result_b, sizeof(uint32_t));
    cudaMalloc(&d_result_c, sizeof(uint32_t));
    cudaMalloc(&d_result_d, sizeof(uint32_t));

    int threads_per_block = 256;
    int number_of_blocks = 256;

    find_nonce<<<number_of_blocks, threads_per_block>>>(d_base_string, base_len, d_target_prefix, prefix_len, d_result_nonce, d_result_a, d_result_b, d_result_c, d_result_d);

    uint32_t result_nonce, result_a, result_b, result_c, result_d;
    cudaMemcpy(&result_nonce, d_result_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result_a, d_result_a, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result_b, d_result_b, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result_c, d_result_c, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&result_d, d_result_d, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("Nonce: %u\n", result_nonce);
    printf("Hash: %08x%08x%08x%08x\n", result_a, result_b, result_c, result_d);

    cudaFree(d_base_string);
    cudaFree(d_target_prefix);
    cudaFree(d_result_nonce);
    cudaFree(d_result_a);
    cudaFree(d_result_b);
    cudaFree(d_result_c);
    cudaFree(d_result_d);

    return 0;
}
