# Mineros PoW de la blockchain

Estos nodos serán los encargados de procesar las tareas generadas por el nodo coordinador aplicando fuerza bruta para resolver el desafío de minado de nuevos bloques de la blockchain. Para calcular el hash que resuelve el desafío utiliza la GPU.

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
