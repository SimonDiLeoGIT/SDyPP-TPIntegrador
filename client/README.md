# Cliente de la blockchain

Este script lo cree para iniciar un proceso que envíe transacciones cada 75 segundos y así generar bloques en la blockchain.

## Instrucciones

1. Instalar dependencias

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Iniciar la aplicación

```sh
flask --app src/server.py run --host 0.0.0.0
```
