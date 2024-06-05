import ast
import pika
import json
import sys
import logging
from flask import Flask, jsonify, request
from datetime import datetime
from model.transaction import Transaction
from plugins.rabbitmq import rabbit_connect
from plugins.redis import redis_connect
from plugins.scheduler import start_cronjob

app = Flask(__name__)
# logging.basicConfig(level=logging.DEBUG)


# Variables globales para mantener las conexiones
rabbit_connection = rabbit_connect()
redis_connection = redis_connect()


def build_block(transactions):
    if (len(transactions) > 0):
        print(f"{datetime.now()}: Building transactions block...",
              file=sys.stdout, flush=True)
        print(f"{transactions}")
    else:
        print(f"{datetime.now()}: There is no transactions",
              file=sys.stdout, flush=True)


def process_transactions():
    # Procesa hasta 50 transacciones o hasta que pase 1 segundo sin que se agreguen transacciones a la queue
    transactions = []
    for method, properties, body in rabbit_connection.consume('transactions', inactivity_timeout=1):
        if method and len(transactions) < 50:
            transaction = ast.literal_eval(body.decode("utf-8"))

            transactions.append(transaction)
            rabbit_connection.basic_ack(method.delivery_tag)
        else:
            build_block(transactions)
            break


# Inicia el cronjob para crear bloques
start_cronjob(process_transactions, 10)


@app.route("/status")
def status():
    return jsonify({
        "status": "200",
        "description": "Coordinator proccess is executing..."
    })


@app.route("/transaction", methods=['POST'])
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
    rabbit_connection.basic_publish(
        exchange='blockchain', routing_key='tx',
        properties=properties,
        body=json.dumps(new_transaction.to_dict()))

    # Almacenar la transacción en redis.
    redis_connection.hset(new_transaction.id,
                          mapping=new_transaction.to_dict())

    return jsonify({
        "status": "200",
        "description": "Transaction registered successfully"
    })
