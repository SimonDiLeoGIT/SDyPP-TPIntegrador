#!/bin/bash

# Obtener la fecha y hora antes de la ejecución
start=$(date +"%s%N")

# Compilar el código CUDA
nvcc find_nonce_gpu.cu -o find_nonce_gpu

# Ejecutar el programa CUDA
output=$(./find_nonce_gpu "0000" "asdasd")

# Obtener la fecha y hora después de la ejecución
end=$(date +"%s%N")

# Calcular la diferencia de tiempo en nanosegundos
elapsed=$((end-start))

# Convertir la diferencia de tiempo a segundos y milisegundos
seconds=$((elapsed/1000000000))
milliseconds=$(( (elapsed/1000000) % 1000 ))

# Imprimir el resultado
echo "$output"
echo "Tiempo transcurrido: $seconds.$milliseconds segundos"