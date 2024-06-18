import ast
import json
import os
import sys
import threading
import time
import random
from datetime import datetime

import pika
from flask import Flask, jsonify, request
from model.block import Block
from pika import exceptions as rabbitmq_exceptions
from plugins.rabbitmq import rabbit_connect
from plugins.redis import redis_connect
from plugins.scheduler import start_cronjob
from plugins.instance_compute import destroy_all_instances, create_multiple_instances, get_active_instance_count
from redis import exceptions as redis_exceptions

app = Flask(__name__)


MAX_RANGE = int(os.environ.get("MAX_RANGE"))
CPU_MINER_INSTANCES = int(os.environ.get("CPU_MINERS_COUNT"))
EXPIRATION_TIME = int(os.environ.get("EXPIRATION_TIME"))
CHECK_POOL_STATUS_INTERVAL = int(os.environ.get("CHECK_POOL_STATUS_INTERVAL"))
CPU_HASH_CHALLENGE = os.environ.get("CPU_HASH_CHALLENGE")

# Variables globales para mantener las conexiones
rabbitmq = rabbit_connect()
redis = redis_connect()


def create_mining_subtasks(block, challenge):
    try:
        gpu_miners_alive = get_gpu_active_nodes()

        if (gpu_miners_alive > 0):
            miners_count = gpu_miners_alive
        else:
            miners_count = get_active_instance_count()
            challenge = CPU_HASH_CHALLENGE

        if (miners_count == 0):
            range_interval = MAX_RANGE
        else:
            range_interval = round(MAX_RANGE/miners_count)
        range_from = 1
        range_to = range_interval

        for i in range(0, miners_count):
            subtask = {
                'from': range_from,
                'to': range_to,
                'challenge': challenge,
                "block": block,
            }

            print(f"subtask: {subtask}", file=sys.stdout, flush=True)

            # Update ranges for the next task
            range_from = range_to + 1
            range_to += range_interval

            properties = pika.BasicProperties(
                delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)

            rabbitmq.basic_publish(
                exchange='blockchain', routing_key='w',
                properties=properties,
                body=json.dumps(subtask))

        return True
    except rabbitmq_exceptions.AMQPError as error:
        print(f"RabbitMQ error: {error}", file=sys.stderr, flush=True)
        return False
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr, flush=True)
        return False


def get_worker_keys(pattern='worker-*'):
    cursor = '0'  # Inicializa el cursor
    worker_keys = []
    while cursor != 0:
        cursor, keys = redis.scan(cursor=cursor, match=pattern)
        worker_keys.extend(keys)
    print(
        f"Claves de todos los workers {worker_keys}", file=sys.stdout, flush=True)
    return worker_keys


def delete_key(key):
    result = redis.delete(key)
    if result == 1:
        print(f"Key '{key}' deleted successfully.",
              file=sys.stderr, flush=True)
    else:
        print(f"Key '{key}' not found.", file=sys.stdout, flush=True)


def check_node_status(node_id):
    try:
        print(f"Checkeando estado de {node_id}", file=sys.stdout, flush=True)
        current_time = int(time.time())
        last_keep_alive = redis.hget(node_id, "last_keep_alive")
        if last_keep_alive:
            last_keep_alive = int(last_keep_alive)
            if current_time - last_keep_alive <= EXPIRATION_TIME:
                print("El nodo ", node_id, " sigue vivo",
                      file=sys.stdout, flush=True)
                return True
            print("Expiró el tiempo del nodo",
                  node_id, file=sys.stdout, flush=True)
            delete_key(node_id)
            return False
        else:
            return False
    except redis_exceptions.RedisError as error:
        print(f"Redis error: {error}", file=sys.stderr, flush=True)
        return False


def get_gpu_active_nodes():
    # Función para obtener la cantidad de nodos activos
    try:
        gpu_active_nodes = 0
        all_nodes = get_worker_keys()
        for node_id in all_nodes:
            if check_node_status(node_id):
                gpu_active_nodes += 1
        return gpu_active_nodes
    except redis_exceptions.RedisError as error:
        print(f"Redis error: {error}", file=sys.stderr, flush=True)
        return 0


def check_pool_status():
    gpu_active_nodes = get_gpu_active_nodes()
    print(
        f"Esta es la cantidad de workers gpu activos: {gpu_active_nodes}", file=sys.stdout, flush=True)

    if (gpu_active_nodes > 0):
        print("Hay mineros GPU activos", file=sys.stdout, flush=True)
        if get_active_instance_count() > 0:
            destroy_all_instances()
    else:
        print("Hay 0 mineros GPU activos", file=sys.stdout, flush=True)
        if get_active_instance_count() == 0:
            create_multiple_instances(CPU_MINER_INSTANCES)
            print(
                f"Se estan creando instancias cpu en la nube: {CPU_MINER_INSTANCES}", file=sys.stdout, flush=True)


start_cronjob(check_pool_status, CHECK_POOL_STATUS_INTERVAL)


@ app.route("/status", methods=['GET'])
def status():
    return jsonify({
        "status": "200",
        "description": "Pool manager proccess is executing..."
    })


@ app.route("/keep-alive", methods=['POST'])
def keep_alive():
    try:
        miner_information = json.loads(request.get_data().decode("utf-8"))

        # Validate that 'node_id' is in the received data
        if "node_id" in miner_information:
            node_id = miner_information["node_id"]

            timestamp = int(time.time())
            redis.hset(node_id, mapping={
                "last_keep_alive": timestamp,
            })

            return jsonify({
                "status": "200",
                "description": "Pool status updated successfully"
            })
        else:
            return jsonify({"status": "error", "message": "node_id not found in the request"}), 400

    except json.JSONDecodeError:
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@ app.route("/register", methods=['GET'])
def register():
    node_id = round(datetime.now().timestamp() + random.randint(0, 1000))
    timestamp = int(time.time())
    redis.hset(f"worker-{node_id}", mapping={
        "last_keep_alive": timestamp,
    })

    return jsonify({
        "status": "200",
        "node_id": f"worker-{node_id}",
    })


def consume_mining_tasks():
    print(" [*] Waiting for messages. To exit press CTRL+C")

    def callback(ch, method, properties, body):
        try:
            mining_task = ast.literal_eval(body.decode("utf-8"))
            challenge = mining_task["challenge"]
            block = mining_task["block"]

            result = create_mining_subtasks(block, challenge)

            if (result):
                rabbitmq.basic_ack(method.delivery_tag)
        except rabbitmq_exceptions.AMQPError as error:
            print(f"RabbitMQ error: {error}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr, flush=True)

    rabbitmq.basic_consume(
        queue="blocks", on_message_callback=callback)
    rabbitmq.start_consuming()


# Iniciar el consumidor al arrancar la aplicación Flask
time.sleep(5)
consumer_thread = threading.Thread(target=consume_mining_tasks)
consumer_thread.start()
