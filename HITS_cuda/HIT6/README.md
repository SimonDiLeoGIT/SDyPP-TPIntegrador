Hit #6 - Longitudes de prefijo en CUDA HASH

Realice mediciones sobre el programa anterior probando diferentes longitudes de prefijo. ¿Cuál es el prefijo más largo que logró encontrar? ¿Cuánto tardo? ¿Cuál es la relación entre la longitud del prefijo a buscar y el tiempo requerido para encontrarlo?


Si bien no hay diferencias de tiempos de ejecucion, esto es simplemente porque el programa que hicimos no corta la ejecucion una vez encuentra un nonce, por lo que se ejecutan todos los threads de todos los bloques.
Con un challenge de "0", se encontraron mas de 1025 nonce posibles (con un tiempo de ejecucion de 1.171 segundos), lo que implica 1 nonce cada 0,0011424390243902 segundos.
Con un challenge de "00" se encontraron unos 74 nonce posibles (con un tiempo de ejecucion de 1.126 segundos), lo que implica 1 nonce cada 0,0152162162162162 segundos.
Con un challenge de "000" se encontraron unos 4 nonce posibles (con un tiempo de ejecucion de 1.130 segundos), lo que implica 1 nonce cada 0,2825 segundos.
Con un challenge de "0000" el programa no encuentra ningun nonce. Esto se debe al rango de valores en el que se esta buscando, cosa que solucionamos en posteriores versiones del programa. Si bien existe la limitacion del tamaño del rango que puede usarse (atada al hardware), podemos ciclar el mismo programa varias veces para cubrir el rango solicitado.