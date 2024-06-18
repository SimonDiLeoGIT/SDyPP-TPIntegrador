import ast
import json
import os
import subprocess
import sys
import threading
import time

import requests
from plugins.scheduler import start_cronjob
from flask import Flask, jsonify
from model.block import Block
from pika import exceptions as rabbitmq_exceptions
from plugins.rabbitmq import rabbit_connect
from utils.check_gpu import check_for_nvidia_smi
from utils.find_nonce import find_nonce_with_prefix

app = Flask(__name__)

# Variables globales para mantener las conexiones
rabbitmq = rabbit_connect()

BLOCKS_COORDINATOR_URL = os.environ.get("BLOCKS_COORDINATOR_URL")
POOL_MANAGER_URL = os.environ.get("POOL_MANAGER_URL")
KEEP_ALIVE_INTERVAL = os.environ.get("KEEP_ALIVE_INTERVAL")
GPU_MAX_RANGE = int(os.environ.get("GPU_MAX_RANGE"))

node_id = None


@ app.route("/status")
def status():
    return jsonify({
        "status": "200",
        "description": "PoW miner is running..."
    })


def send_register():
    try:
        print("Registering with pool manager...",
              file=sys.stdout, flush=True)
        global node_id
        while node_id is None:
            response = requests.get(f"{POOL_MANAGER_URL}/register")
            # Lanza una excepción si la respuesta tiene un código de estado HTTP de error
            response.raise_for_status()

            # Intenta analizar la respuesta como JSON
            data = ast.literal_eval(response.text)

            print(f"data: {data.get('node_id')}",
                  file=sys.stdout, flush=True)

            # Extrae el valor de 'node_id'
            node_id = data.get('node_id')

            if node_id is not None:
                print(
                    f"GPU worker {node_id} registered succesfully.", file=sys.stdout, flush=True)
            else:
                print("Waiting for pool manager to be ready...",
                      file=sys.stdout, flush=True)

    except requests.exceptions.RequestException as e:
        print(f"HTTP request error: {e}", file=sys.stderr, flush=True)
    except ValueError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr, flush=True)


def send_keep_alive():
    try:
        body = {
            "node_id": node_id,
        }

        response = requests.post(
            f"{POOL_MANAGER_URL}/keep-alive", json.dumps(body))
        print(response.text, file=sys.stdout, flush=True)

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr, flush=True)


def consume_tasks():
    print(" [*] Waiting for messages. To exit press CTRL+C")

    def callback(ch, method, properties, body):
        try:
            task = json.loads(body)

            challenge = task["challenge"]
            block = task["block"]
            from_range = task["from"]
            to_range = task["to"]

            print(f"to_range: {to_range} type: {type(to_range)}",
                  file=sys.stdout, flush=True)
            print(f"from_range: {from_range} type: {type(from_range)}",
                  file=sys.stdout, flush=True)

            block_hash = ""
            nonce = 0
            block = Block(block["data"], block["timestamp"],
                          block_hash, block["previous_hash"], nonce, block["index"])

            # Obtiene el hash del contenido del bloque que servirá como entrada al minero
            block_content_hash = block.get_block_content_hash()

            if gpu_available:
                print("GPU available")
                current_dir = os.getcwd()

                # Define relative paths
                src_dir = os.path.join(current_dir, "src/utils/cuda")
                src_file = "find_nonce_gpu.cu"
                output_file = "find_nonce_gpu"
                result_file = "output.json"

                if not os.path.isdir(src_dir):
                    return

                # Create the full paths
                src_path = os.path.join(src_dir, src_file)
                output_path = os.path.join(src_dir, output_file)
                # result_path = os.path.join(src_dir, result_file)
                result_path = os.path.join(current_dir, result_file)

                # Crea vacío el archivo de resultado
                with open(result_path, "w") as f:
                    f.truncate(0)

                gpu_from = from_range
                gpu_to = gpu_from + GPU_MAX_RANGE
                nonce_found = False

                if not os.path.isfile(output_path):
                    # Call nvcc to compile the CUDA file if the output file does not exist
                    subprocess.call(["nvcc", src_path, "-o", output_path])

                # Invocacion del cuda
                while (not nonce_found) and (gpu_to <= to_range):
                    subprocess.call(
                        [output_path, str(gpu_from), str(gpu_to), challenge, block_content_hash], stdout=subprocess.DEVNULL)

                    file = open(result_path, "r")
                    result = file.readlines()
                    result = ast.literal_eval(result[0])
                    block_hash = result["block_hash"]
                    nonce = result["nonce"]
                    if (nonce > 0) and (block_hash != ""):
                        nonce_found = True
                    else:
                        gpu_from = gpu_to + 1
                        gpu_to += GPU_MAX_RANGE

                        if (gpu_to > to_range):
                            gpu_to = to_range
            else:
                print("GPU no available")
                nonce, block_hash = find_nonce_with_prefix(
                    challenge, block_content_hash, 0, 1000000)

            print(f"block_hash: {block_hash}", file=sys.stdout, flush=True)
            print(f"nonce: {nonce}", file=sys.stdout, flush=True)

            # Actualiza el bloque con los valores calculador por el minero_cuda
            block.hash = block_hash
            block.nonce = nonce

            # Envía el bloque con los datos de hash y nonce al coordinador para que lo valide
            response = requests.post(
                BLOCKS_COORDINATOR_URL, json.dumps(block.to_dict()))
            print(response.text, file=sys.stdout, flush=True)
            rabbitmq.basic_ack(method.delivery_tag)

        except rabbitmq_exceptions.AMQPError as error:
            print(f"RabbitMQ error: {error}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr, flush=True)

    rabbitmq.basic_consume(
        queue="workers", on_message_callback=callback)
    rabbitmq.start_consuming()


gpu_available = check_for_nvidia_smi()
if gpu_available:
    # Iniciar el cronjob para emitir los keep-alive
    send_register()
    start_cronjob(send_keep_alive, int(KEEP_ALIVE_INTERVAL))

# Iniciar el consumidor al arrancar la aplicación Flask
consumer_thread = threading.Thread(target=consume_tasks)
consumer_thread.start()
