#include <iostream>
#include <cstdio>
#include <curand_kernel.h>
#include <chrono>
#include "md5.cu"

// Función de kernel para calcular el hash MD5
__global__ void calculateMD5(const char *input, size_t input_length, const char *target_prefix,
                             size_t target_length, uint32_t *a, uint32_t *b, uint32_t *c, uint32_t *d)
{
  // Obtener el índice del hilo
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Semilla aleatoria basada en el tiempo y el índice del hilo
  curandState_t state;
  curand_init(clock64() + idx, 0, 0, &state);

  // Buffer para almacenar la cadena concatenada
  char buffer[64]; // Longitud máxima del mensaje para MD5

  // Inicializar la parte conocida de la cadena con el objetivo
  for (size_t i = 0; i < input_length; ++i)
  {
    buffer[i] = input[i];
  }
  for (size_t i = 0; i < target_length; ++i)
  {
    buffer[input_length + i] = target_prefix[i];
  }
  buffer[input_length + target_length] = '\0'; // Asegurar que la cadena esté terminada correctamente

  // Inicializar el hash MD5 con la cadena conocida
  md5Hash((unsigned char *)buffer, input_length + target_length, a, b, c, d);

  // Iterar hasta encontrar un hash con el prefijo objetivo
  while (true)
  {
    // Generar un número aleatorio
    unsigned int random_number = curand(&state);

    // Convertir el número aleatorio en una cadena
    char random_str[16]; // Suficiente para representar un uint32_t
    int pos = 0;
    do
    {
      random_str[pos++] = '0' + (random_number % 10);
      random_number /= 10;
    } while (random_number > 0);
    random_str[pos] = '\0';

    // Concatenar el número aleatorio con la cadena conocida
    for (size_t i = 0; i < pos; ++i)
    {
      buffer[input_length + target_length + i] = random_str[pos - 1 - i];
    }
    buffer[input_length + target_length + pos] = '\0'; // Asegurar que la cadena esté terminada correctamente

    // Calcular el hash MD5 de la cadena resultante
    md5Hash((unsigned char *)buffer, input_length + target_length + pos, a, b, c, d);

    // Verificar si el hash comienza con el prefijo objetivo
    bool match = true;
    for (size_t i = 0; i < target_length; ++i)
    {
      if (buffer[i] != target_prefix[i])
      {
        match = false;
        break;
      }
    }
    if (match)
    {
      break; // Se encontró el hash deseado
    }
  }
}

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    std::cerr << "Uso: " << argv[0] << " <cadena> <prefijo_objetivo>" << std::endl;
    return 1;
  }

  const char *input = argv[1];         // Cadena proporcionada por el usuario
  const char *target_prefix = argv[2]; // Prefijo objetivo del hash

  size_t input_length = strlen(input);
  size_t target_length = strlen(target_prefix);

  // Variables para almacenar los resultados del hash MD5
  uint32_t a, b, c, d;

  // Variables en el dispositivo para almacenar los resultados
  uint32_t *dev_a, *dev_b, *dev_c, *dev_d;
  cudaMalloc((void **)&dev_a, sizeof(uint32_t));
  cudaMalloc((void **)&dev_b, sizeof(uint32_t));
  cudaMalloc((void **)&dev_c, sizeof(uint32_t));
  cudaMalloc((void **)&dev_d, sizeof(uint32_t));

  // Iniciar el temporizador
  auto start_time = std::chrono::high_resolution_clock::now();

  // Llama al kernel para calcular el hash MD5
  calculateMD5<<<1, 1024>>>(input, input_length, target_prefix, target_length, dev_a, dev_b, dev_c, dev_d);

  // Detener el temporizador
  auto end_time = std::chrono::high_resolution_clock::now();

  // Copia los resultados de vuelta desde el dispositivo
  cudaMemcpy(&a, dev_a, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&b, dev_b, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&c, dev_c, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&d, dev_d, sizeof(uint32_t), cudaMemcpyDeviceToHost);

  // Calcular el tiempo transcurrido
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

  // Convertir a milisegundos
  double duration_ms = duration.count() / 1000.0;

  // Imprime el hash calculado y el tiempo de ejecución
  std::cout << "Hash MD5 de \"" << input << "\" con prefijo objetivo \"" << target_prefix << "\": ";
  std::cout << std::hex << a << b << c << d << std::endl;
  std::cout << "Tiempo de ejecución: " << duration_ms << " milisegundos" << std::endl;

  // Libera la memoria en el dispositivo
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  cudaFree(dev_d);

  return 0;
}