import ast
import pika
import json
import sys
import time
import logging
from flask import Flask, jsonify, request
from datetime import datetime
from model.transaction import Transaction
from model.block import Block
from plugins.rabbitmq import rabbit_connect
from plugins.redis import redis_connect
from plugins.scheduler import start_cronjob

app = Flask(__name__)
# logging.basicConfig(level=logging.DEBUG)


# Variables globales para mantener las conexiones
rabbitmq = rabbit_connect()
redis = redis_connect()


def build_block(transactions):
    if (len(transactions) > 0):
        # Obtengo el id del último bloque de la blockchain
        last_block_hash = redis.zrange(
            'blockchain', -1, -1, withscores=True)

        last_index = redis.zcount('blockchain', '-inf', '+inf')

        if (len(last_block_hash) == 0):
            # Si last_block = [] se crea el bloque genesis
            previous_hash = 0
        else:
            # Obtengo el último bloque de la blochain
            previous_hash = last_block_hash[0][0]

        print(f"{datetime.now()}: Building transactions block...",
              file=sys.stdout, flush=True)

        # TODO => Verificar si están todos los datos necesarios
        block = {
            'index': last_index,
            'previous_hash': previous_hash,
            'data': transactions,
            "timestamp": f"{round(time.time())}",
            'nonce': 0,  # Este valor lo calculan los mineros
            # Este valor lo completarán los mineros con el siguiente cálculo md5(index+timestamp+data+previous_hash+nonce)
            'hash': "",
        }

        new_block = Block(
            block["data"], block["timestamp"], block["hash"], block["previous_hash"], block["nonce"], block["index"])

        properties = pika.BasicProperties(
            delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)

        rabbitmq.basic_publish(
            exchange='workers', routing_key='block',
            properties=properties,
            body=json.dumps({"challengue": "000", "block": new_block.to_dict()}))

        print(
            f"{datetime.now()}: Block {new_block.index} [{new_block.previous_hash}] created ...")

    else:
        print(f"{datetime.now()}: There is no transactions",
              file=sys.stdout, flush=True)


def process_transactions():
    # Procesa hasta 50 transacciones o hasta que pase 1 segundo sin que se agreguen transacciones a la queue
    transactions = []
    for method, properties, body in rabbitmq.consume('transactions', inactivity_timeout=1):
        if method and len(transactions) < 50:
            transaction = ast.literal_eval(body.decode("utf-8"))
            transactions.append(transaction)
            rabbitmq.basic_ack(method.delivery_tag)
        else:
            build_block(transactions)
            break


# Inicia el cronjob para crear bloques
start_cronjob(process_transactions, 60)


@ app.route("/status")
def status():
    return jsonify({
        "status": "200",
        "description": "Coordinator proccess is executing..."
    })


@ app.route("/transaction", methods=['POST'])
def registerTransaction():
    transaction = ast.literal_eval(request.get_data().decode("utf-8"))

    sender = transaction["sender"]
    receiver = transaction["receiver"]
    amount = transaction["amount"]
    signature = transaction["signature"]
    timestamp = f"{datetime.now()}"
    transaction_id = f"tx:{timestamp}:{sender}"

    new_transaction = Transaction(
        transaction_id, sender, receiver, amount, signature, timestamp)

    # Publico la transacción en la cola de rabbit.
    properties = pika.BasicProperties(
        delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)
    rabbitmq.basic_publish(
        exchange='blockchain', routing_key='tx',
        properties=properties,
        body=json.dumps(new_transaction.to_dict()))

    # Almacenar la transacción en redis.
    redis.hset(new_transaction.id,
               mapping=new_transaction.to_dict())

    return jsonify({
        "status": "200",
        "description": "Transaction registered successfully"
    })


@ app.route("/result", methods=['POST'])
def validateBlock():
    block = ast.literal_eval(request.get_data().decode("utf-8"))
    print(block)

    timestamp = block["timestamp"]
    block_hash = block["hash"]
    previous_hash = block["previous_hash"]
    data = block["data"]
    index = block["index"]
    nonce = block["nonce"]

    new_block = Block(
        block["data"], block["timestamp"], block["hash"], block["previous_hash"], block["nonce"], block["index"])

    # TODO => Validar el bloque
    # Verificando el hash md5(index+timestamp+data+previous_hash+nonce)

    # TODO => Verificar si existe en redis. Si no existe almacenarlo. Si ya existe descarto está request, porque ya un minero completo la tarea antes
    # Guardo el indice del nuevo bloque en el sorted set
    redis.zadd('blockchain', {new_block.hash: time.time()})
    # Guardo el bloque en la blockchain, asociandoló con el bloque anterior
    block_id = f"block:{new_block.previous_hash}"
    redis.hset(block_id, mapping=new_block.to_dict())
