# Instrucciones para correr el TP de manera local


### Levantar la blockchain

1. Iniciar la aplicación.

```sh
sh build.sh
```


### Levantar los mineros aparte

#### Se pueden levantar N cantidad de mineros.

1. Dirigirse a la carpeta del minero

```sh
cd minero
```
2. Instalan las dependencias:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
3.Crear .env con los siguientes valores dentro de la carpeta del minero

```sh
RABBITMQ_USER=admin
RABBITMQ_PASSWORD=password
RABBITMQ_HOST=localhost
BLOCKS_COORDINATOR_URL=http://localhost:5001/result
POOL_MANAGER_URL=http://localhost:5002/result
KEEP_ALIVE_INTERVAL=5
GPU_MAX_RANGE=80000
```

3. Iniciar la aplicación:

```sh
flask --app src/server.py run --host 0.0.0.0
```
4. Enviar transacciones para empezar los procesos:

```sh
curl -X POST \
  http://localhost:5001/ \
  -H 'Content-Type: application/json' \
  -d '{
    "sender": "9b22add3e341cbada2c289d6d6e28b70",
    "receiver": "259a2d1f68fef2c2b38eddd9b4eb2f10",
    "amount": 5
  }'
```
