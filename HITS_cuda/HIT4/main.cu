#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include "md5.cu"

__global__ void compute_md5(const uint8_t* message, size_t length, uint8_t* digest) {
    cuda_md5(message, length, digest);
}

void print_md5(uint8_t* digest) {
    for (int i = 0; i < 16; i++) {
        printf("%02x", digest[i]);
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <string>" << std::endl;
        return 1;
    }

    const char* input = argv[1];
    size_t length = strlen(input);

    uint8_t* d_message;
    uint8_t* d_digest;
    uint8_t h_digest[16];

    cudaMalloc(&d_message, length);
    cudaMalloc(&d_digest, 16);

    cudaMemcpy(d_message, input, length, cudaMemcpyHostToDevice);

    compute_md5<<<1, 1>>>(d_message, length, d_digest);

    cudaMemcpy(h_digest, d_digest, 16, cudaMemcpyDeviceToHost);

    print_md5(h_digest);

    cudaFree(d_message);
    cudaFree(d_digest);

    return 0;
}