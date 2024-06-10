import ast
import json
import os
import sys
import threading
import time
import requests

from flask import Flask, jsonify
from pika import exceptions as rabbitmq_exceptions
from model.block import Block
from plugins.rabbitmq import rabbit_connect
from utils.check_gpu import check_for_nvidia_smi
from utils.find_nonce import find_nonce_with_prefix

app = Flask(__name__)


# Variables globales para mantener las conexiones
rabbitmq = rabbit_connect()
coordinator_url = os.environ.get("COORDINATOR_URL")


@ app.route("/status")
def status():
    return jsonify({
        "status": "200",
        "description": "Coordinator proccess is executing..."
    })


def consume_tasks():
    print(" [*] Waiting for messages. To exit press CTRL+C")

    def callback(ch, method, properties, body):
        try:
            task = ast.literal_eval(body.decode("utf-8"))
            challenge = task["challenge"]
            block = task["block"]

            print(f"Challengue: {challenge}", file=sys.stdout, flush=True)
            print(f"Block: {block}", file=sys.stdout, flush=True)

            block_hash = ""
            nonce = 0

            block = Block(block["data"], block["timestamp"],
                          block_hash, block["previous_hash"], nonce, block["index"])

            if gpu_available:
                print("GPU available")
                # TODO: Invocar al minero_cuda para que calcule el hash y el nonce
                block_hash = challenge + "XXXXXXXX"
                nonce = 23
            else:
                print("GPU no available")
                block_content = block.get_block_content_as_string()
                nonce, block_hash = find_nonce_with_prefix(
                    challenge, block_content, 0, 1000000)

            print(f"block_hash: {block_hash}", file=sys.stdout, flush=True)
            print(f"nonce: {nonce}", file=sys.stdout, flush=True)

            # Actualiza el bloque con los valores calculador por el minero_cuda
            block.hash = block_hash
            block.nonce = nonce

            # Envía el bloque con los datos de hash y nonce al coordinador para que lo valide
            response = requests.post(
                coordinator_url, json.dumps(block.to_dict()))
            if response.status_code == 200:
                response_json = response.json()
                print(response_json)
            else:
                print(
                    f"Failed to send data. Status code: {response.status_code}")
                print(response.text)

            ch.basic_ack(delivery_tag=method.delivery_tag)
        except rabbitmq_exceptions.AMQPError as error:
            print(f"RabbitMQ error: {error}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr, flush=True)

    rabbitmq.basic_consume(
        queue='blocks', on_message_callback=callback,)
    rabbitmq.start_consuming()


# Iniciar el consumidor al arrancar la aplicación Flask
time.sleep(5)
gpu_available = check_for_nvidia_smi()
consumer_thread = threading.Thread(target=consume_tasks)
consumer_thread.start()
