# Nodo coordinador de la blockchain

## Instrucciones

1. Instalar dependencias

```sh
 python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Iniciar la aplicación

```sh
# flask --app src/server.py run --host 0.0.0.0
docker build -t coordinator:latest .
docker-compose up
```

Durante el desarrollo directamente usaba esta serie de comandos. Así era más rapido a la hora de modificar el código y volver a iniciar todos los contenedores.

```sh
docker-compose down ; docker image rm coordinator:latest ; docker build -t coordinator:latest . ; docker-compose up -d
```
