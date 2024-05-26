#include <iostream>
#include <cstring>
#include "md5.cu"

__global__ void calculateMD5(const char *input, size_t length, uint32_t *a, uint32_t *b, uint32_t *c, uint32_t *d)
{
  md5Hash((unsigned char *)input, length, a, b, c, d);
}

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cerr << "Uso: " << argv[0] << " <cadena>" << std::endl;
    return 1;
  }

  // Obtén el string a hashear desde los argumentos de la línea de comandos
  const char *input = argv[1];
  size_t length = std::strlen(input);

  // Variables para almacenar los resultados del hash MD5
  uint32_t a, b, c, d;

  // Variables en el dispositivo para almacenar los resultados
  uint32_t *dev_a, *dev_b, *dev_c, *dev_d;
  cudaMalloc((void **)&dev_a, sizeof(uint32_t));
  cudaMalloc((void **)&dev_b, sizeof(uint32_t));
  cudaMalloc((void **)&dev_c, sizeof(uint32_t));
  cudaMalloc((void **)&dev_d, sizeof(uint32_t));

  // Llama al kernel para calcular el hash MD5
  calculateMD5<<<1, 1>>>(input, length, dev_a, dev_b, dev_c, dev_d);

  // Copia los resultados de vuelta desde el dispositivo
  cudaMemcpy(&a, dev_a, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&b, dev_b, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&c, dev_c, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&d, dev_d, sizeof(uint32_t), cudaMemcpyDeviceToHost);

  // Imprime el hash calculado
  std::cout << "Hash MD5 de \"" << input << "\": ";
  std::cout << std::hex << a << b << c << d << std::endl;

  // Libera la memoria en el dispositivo
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  cudaFree(dev_d);

  return 0;
}