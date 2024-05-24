#include <stdio.h>

__global__ void helloFromGPU()
{
  printf("Hola, mundo desde GPU!\n");
}

// Macro para verificar errores de CUDA
#define gpuErrorCheck(call)
do
{
  cudaError_t gpuErr = call;
  if (cudaSuccess != gpuErr)
  {
    printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(gpuErr));
    exit(1);
  }
} while (0)

    int
    main()
{
  printf("Iniciando programa...\n");

  // Lanzar el kernel y verificar errores
  helloFromGPU<<<1, 1>>>();
  gpuErrorCheck(cudaGetLastError());      // Verificar errores después del lanzamiento del kernel
  gpuErrorCheck(cudaDeviceSynchronize()); // Esperar a que el kernel termine y verificar errores

  printf("Finalizando programa...\n");

  return 0;
}