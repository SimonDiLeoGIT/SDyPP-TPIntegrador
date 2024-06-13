import subprocess
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
rabbitmq, queue_name = rabbit_connect()
coordinator_url = os.environ.get("COORDINATOR_URL")


@ app.route("/status")
def status():
    return jsonify({
        "status": "200",
        "description": "PoW miner is running..."
    })


def consume_tasks():
    print(" [*] Waiting for messages. To exit press CTRL+C")

    def callback(ch, method, properties, body):
        try:
            task = ast.literal_eval(body.decode("utf-8"))
            challenge = task["challenge"]
            block = task["block"]
            print(f"Block: {block}", file=sys.stdout, flush=True)

            block_hash = ""
            nonce = 0
            block = Block(block["data"], block["timestamp"],
                          block_hash, block["previous_hash"], nonce, block["index"])
            block_content = block.get_block_content_as_string()

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
                result_path = os.path.join(src_dir, result_file)
                result_path = os.path.join(current_dir, result_file)

                if not os.path.isfile(output_path):
                    # Call nvcc to compile the CUDA file if the output file does not exist
                    subprocess.call(["nvcc", src_path, "-o", output_path])

                block_content = block.get_block_content_as_string()
                subprocess.call(
                    [output_path, "1", "10000", challenge, block_content], stdout=subprocess.DEVNULL)

                file = open(result_path, "r")
                result = file.readlines()
                result = ast.literal_eval(result[0])
                block_hash = result["block_hash"]
                nonce = result["nonce"]
            else:
                print("GPU no available")
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
            print(response.text, file=sys.stdout, flush=True)

        except rabbitmq_exceptions.AMQPError as error:
            print(f"RabbitMQ error: {error}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr, flush=True)

    rabbitmq.basic_consume(
        queue=queue_name, on_message_callback=callback, auto_ack=True)
    rabbitmq.start_consuming()


# Iniciar el consumidor al arrancar la aplicación Flask
time.sleep(5)
gpu_available = check_for_nvidia_smi()
consumer_thread = threading.Thread(target=consume_tasks)
consumer_thread.start()
