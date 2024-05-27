#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include "md5.cu"

__device__ int custom_strlen(const char* str) {
    int len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

__device__ void custom_itoa(int num, char* str) {
    int i = 0;
    int is_negative = 0;

    if (num == 0) {
        str[i++] = '0';
        str[i] = '\0';
        return;
    }

    if (num < 0) {
        is_negative = 1;
        num = -num;
    }

    while (num != 0) {
        int rem = num % 10;
        str[i++] = rem + '0';
        num = num / 10;
    }

    if (is_negative) {
        str[i++] = '-';
    }

    str[i] = '\0';

    for (int j = 0, k = i - 1; j < k; j++, k--) {
        char temp = str[j];
        str[j] = str[k];
        str[k] = temp;
    }
}

__device__ int custom_strncmp(const char* str1, const char* str2, int n) {
    for (int i = 0; i < n; i++) {
        if (str1[i] != str2[i] || str1[i] == '\0' || str2[i] == '\0') {
            return str1[i] - str2[i];
        }
    }
    return 0;
}

__device__ void custom_snprintf(char* buffer, int buffer_size, const char* format, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    if (buffer_size < 33) return;
    for (int i = 0; i < 32; i++) {
        buffer[i] = '0';
    }
    buffer[32] = '\0';

    for (int i = 0; i < 8; i++) {
        buffer[i] = "0123456789abcdef"[(a >> (28 - 4 * i)) & 0xF];
        buffer[8 + i] = "0123456789abcdef"[(b >> (28 - 4 * i)) & 0xF];
        buffer[16 + i] = "0123456789abcdef"[(c >> (28 - 4 * i)) & 0xF];
        buffer[24 + i] = "0123456789abcdef"[(d >> (28 - 4 * i)) & 0xF];
    }
}

__device__ void concatenateStrings(unsigned char* combined_string, const unsigned char* input_string, int string_length, unsigned int number) {
    char number_str[12];
    custom_itoa(number, number_str);

    for (int i = 0; i < string_length; i++) {
        combined_string[i] = input_string[i];
    }

    int number_length = custom_strlen(number_str);
    for (int i = 0; i < number_length; i++) {
        combined_string[string_length + i] = number_str[i];
    }
    combined_string[string_length + number_length] = '\0';
}

__global__ void findHashPrefixKernel(unsigned char* input_string, int string_length, const char* target_prefix, char* result_hash, int* result_number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int number = idx;

    while (1) {
        unsigned char combined_string[256];
        concatenateStrings(combined_string, input_string, string_length, number);

        uint32_t a, b, c, d;
        md5Hash(combined_string, custom_strlen((char*)combined_string), &a, &b, &c, &d);

        char hash_str[33];
        custom_snprintf(hash_str, sizeof(hash_str), "%08x%08x%08x%08x", a, b, c, d);

        if (custom_strncmp(hash_str, target_prefix, custom_strlen(target_prefix)) == 0) {
            printf("Found match: %s for number %u\n", hash_str, number);
            for (int i = 0; i < 33; i++) {
                result_hash[i] = hash_str[i];
            }
            *result_number = number;
            return;
        }

        number += gridDim.x * blockDim.x;
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_string> <target_prefix>\n", argv[0]);
        return 1;
    }

    const char* input_string = argv[1];
    const char* target_prefix = argv[2];
    int string_length = strlen(input_string);

    unsigned char* d_input_string;
    char* d_result_hash;
    int* d_result_number;

    checkCudaError(cudaMalloc((void**)&d_input_string, (string_length + 1) * sizeof(unsigned char)), "Allocating d_input_string");
    checkCudaError(cudaMemcpy(d_input_string, input_string, (string_length + 1) * sizeof(unsigned char), cudaMemcpyHostToDevice), "Copying input_string to d_input_string");

    checkCudaError(cudaMalloc((void**)&d_result_hash, 33 * sizeof(char)), "Allocating d_result_hash");
    checkCudaError(cudaMalloc((void**)&d_result_number, sizeof(int)), "Allocating d_result_number");

    findHashPrefixKernel<<<256, 256>>>(d_input_string, string_length, target_prefix, d_result_hash, d_result_number);

    checkCudaError(cudaPeekAtLastError(), "Launching kernel");
    checkCudaError(cudaDeviceSynchronize(), "Synchronizing after kernel");

    char result_hash[33];
    int result_number;
    checkCudaError(cudaMemcpy(result_hash, d_result_hash, 33 * sizeof(char), cudaMemcpyDeviceToHost), "Copying d_result_hash to result_hash");
    checkCudaError(cudaMemcpy(&result_number, d_result_number, sizeof(int), cudaMemcpyDeviceToHost), "Copying d_result_number to result_number");

    cudaFree(d_input_string);
    cudaFree(d_result_hash);
    cudaFree(d_result_number);

    printf("Found hash: %s\n", result_hash);
    printf("With number: %d\n", result_number);

    return 0;
}
