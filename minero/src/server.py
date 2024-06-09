import ast
import sys
import time
import threading
import requests
import os
from plugins.rabbitmq import rabbit_connect
from flask import Flask, jsonify
from model.block import Block
import json


app = Flask(__name__)
# logging.basicConfig(level=logging.DEBUG)


# Variables globales para mantener las conexiones
rabbitmq = rabbit_connect()
coordinator_url = os.environ.get("COORDINATOR_URL")


@ app.route("/status")
def status():
    return jsonify({
        "status": "200",
        "description": "Coordinator proccess is executing..."
    })


# @ app.route("/result", methods=['POST'])
# def validateBlock():
#     try:
#         block = ast.literal_eval(request.get_data().decode("utf-8"))

#         timestamp = block["timestamp"]
#         block_hash = block["hash"]
#         previous_hash = block["previous_hash"]
#         data = block["data"]
#         index = block["index"]
#         nonce = block["nonce"]

#         new_block = Block(
#             data, timestamp, block_hash, previous_hash, nonce, index)

#         # TODO => Validar el bloque
#         new_block.validate()

#         # TODO => Verificar si existe en redis. Si no existe almacenarlo. Si ya existe descarto está request, porque ya un minero completo la tarea antes
#         # Guardo el indice del nuevo bloque en el sorted set
#         redis.zadd('blockchain', {new_block.hash: time.time()})
#         # Guardo el bloque en la blockchain, asociandoló con el bloque anterior
#         block_id = f"block:{new_block.previous_hash}"
#         redis.hset(block_id, mapping=new_block.to_dict())

#         return jsonify({
#             "status": "200",
#             "description": f"Block {new_block.index} created",
#             "block_data": new_block.to_dict()
#         })

#     except redis_exceptions.RedisError as error:
#         print(f"Redis error: {error}", file=sys.stderr, flush=True)
#         return jsonify({
#             "status": "500",
#             "description": "Internal server error"
#         })
#     except Exception as e:
#         print(f"Unexpected error: {e}", file=sys.stderr, flush=True)
#         return jsonify({
#             "status": "500",
#             "description": "Internal server error"
#         })

def consume_tasks():
    print(" [*] Waiting for messages. To exit press CTRL+C")

    def callback(ch, method, properties, body):
        task = ast.literal_eval(body.decode("utf-8"))
        challenge = task["challenge"]
        block = task["block"]

        print(f"Challengue: {challenge}", file=sys.stdout, flush=True)
        print(f"Block: {block}", file=sys.stdout, flush=True)

        block_hash = ""
        nonce = 0

        block = Block(block["data"], block["timestamp"],
                      block_hash, block["previous_hash"], nonce, block["index"])

        # TODO => Invocar al minero_cuda para que calcule el hash y el nonce
        block_content = block.get_block_content_as_string()
        block_hash = challenge + "XXXXXXXX"
        nonce = 23

        print(block_content, file=sys.stdout, flush=True)
        # Actualiza el bloque con los valores calculador por el minero_cuda
        block.hash = block_hash
        block.nonce = nonce

        # Envía el bloque con los datos de hash y nonce al coordinador para que lo valide
        response = requests.post(coordinator_url, json.dumps(block.to_dict()))
        if response.status_code == 200:
            response_json = response.json()
            print(response_json)
        else:
            print(f"Failed to send data. Status code: {response.status_code}")
            print(response.text)

        ch.basic_ack(delivery_tag=method.delivery_tag)

    rabbitmq.basic_consume(
        queue='blocks', on_message_callback=callback)
    rabbitmq.start_consuming()


# Iniciar el consumidor al arrancar la aplicación Flask
time.sleep(5)
consumer_thread = threading.Thread(target=consume_tasks)
consumer_thread.start()
