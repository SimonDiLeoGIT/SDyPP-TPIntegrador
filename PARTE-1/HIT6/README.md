Hit #6 - Longitudes de prefijo en CUDA HASH

Realice mediciones sobre el programa anterior probando diferentes longitudes de prefijo. ¿Cuál es el prefijo más largo que logró encontrar? ¿Cuánto tardo? ¿Cuál es la relación entre la longitud del prefijo a buscar y el tiempo requerido para encontrarlo?


- Compilación y ejecución:
```sh
nvcc -o brute_md5 brute_md5_wtime.cu -std=c++11 -arch=sm_35
./brute_md5 '<cadena> <prefijo_objetivo>'
```