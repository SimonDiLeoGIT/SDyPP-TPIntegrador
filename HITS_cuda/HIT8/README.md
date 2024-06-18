Hit #8 - HASH GPU vs HASH CPU

Volvamos al programa que hizo en Hit #7, implemente el mismo pero usando solo CPU, es decir, quite CUDA del mismo y haga un programa tradicional que resuelva el mismo problema. Elabore una batería de tests (parámetros de entrada) y ejecutarlos en GPU y en CPU, elabore una comparativa de los resultados obtenidos.

- Worker CPU - Ejecución:
Ir a la carpeta "minero-cpu" y ejecutar el comando:
```sh
python3 find_nonce.py <challenge> <string> <from> <to> 
```
Reemplaze los campos con los valores deseados


- Worker GPU - Compilación y Ejecución:
Ir a la carpeta "minero-gpu" y ejecutar el comando:
```sh
python3 server.py <from> <to> <challenge> <string>
```
Reemplaze los campos con los valores deseados


En el worker GPU se obtuvieron los siguientes resultados:
* Challenge: "0", Cant. nonces: 1025, Tiempo de ejecucion 0.19 s, nonce/s: 1,853658536585366e-4 s.
* Challenge: "00", Cant. nonces: 74, Tiempo de ejecucion 0.19 s, nonce/s: 0,0025675675675676 s.
* Challenge: "000", Cant. nonces: 4, Tiempo de ejecucion 0.19 s, nonce/s: 0,2825 s.
* Challenge: "0000", Cant. nonces: 1, Tiempo de ejecucion 0.19 s, nonce/s: 0.19 s.
* Challenge: "00000", Cant. nonces: 1, Tiempo de ejecucion 1.35 s, nonce/s: 1.35 s,.
* Challenge: "000000", Cant. nonces: 1, Tiempo de ejecucion 16.22 s, nonce/s: 16.22 s.
* Challenge: "0000000", Cant. nonces: 1, Tiempo de ejecucion 593.22 s, nonce/s: 593.22 s (9,9 minutos).

Por su parte, el minero CPU (escrito en python) termina la ejecucion del programa apenas encuentra un nonce, por lo que el tiempo de ejecucion es el mismo tiempo que nonce/s. Dicho esto, los resultados fueron los siguientes:
* Challenge: "0", nonce/s: 0.000031 s.
* Challenge: "00", nonce/s:  0.000252 s.
* Challenge: "000", nonce/s: 0.006380 s.
* Challenge: "0000", nonce/s: 0.015636 s.
* Challenge: "00000", nonce/s: 0.488750 s.
* Challenge: "000000", nonce/s: 6.958972 s.
* Challenge: "0000000", nonce/s: 250.350349 s (4,17 minutos).

Creemos que estos resultados se deben a un overhead de tiempo de ejecucion en el codigo "server.py" y que si se tratara de encontrar nonce con challenges mas grandes, en rangos mas amplios, esta diferencia se veria invertida (el minero de gpu tardaria menos que el de gpu). Sin embargo, estas pruebas tomarian una gran cantidad de tiempo.