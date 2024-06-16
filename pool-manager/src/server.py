import ast
import pika
import json
import sys
import threading
import time
import os
from redis import exceptions as redis_exceptions
from pika import exceptions as rabbitmq_exceptions
from flask import Flask, jsonify, request
from datetime import datetime
from model.block import Block
from plugins.rabbitmq import rabbit_connect
from plugins.redis import redis_connect
from plugins.scheduler import start_cronjob

app = Flask(__name__)


MAX_RANGE = int(os.environ.get("MAX_RANGE"))
CPU_MINER_INSTANCES = int(os.environ.get("CPU_MINERS_COUNT"))
EXPIRATION_TIME = int(os.environ.get("EXPIRATION_TIME"))

# Variables globales para mantener las conexiones
rabbitmq = rabbit_connect()
redis = redis_connect()


def create_mining_subtasks(block, challenge):
    try:
        gpu_miners_alive = get_gpu_active_nodes()

        if (gpu_miners_alive > 0):
            miners_count = gpu_miners_alive
        else:
            miners_count = CPU_MINER_INSTANCES

        range_interval = round(MAX_RANGE/miners_count)
        range_from = 1
        range_to = range_interval

        for i in range(0, miners_count):
            subtask = {
                'from': range_from,
                'previous_hash': range_to,
                'challenge': challenge,
                "block": block,
            }

            # Update ranges for the next task
            range_from = + range_to
            range_to += range_interval

            properties = pika.BasicProperties(
                delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE)

            rabbitmq.basic_publish(
                exchange='blockchain', routing_key='mt',
                properties=properties,
                body=json.dumps({"challenge": challenge, "mining_task": subtask.to_dict()}))

        return True
    except rabbitmq_exceptions.AMQPError as error:
        print(f"RabbitMQ error: {error}", file=sys.stderr, flush=True)
        return False
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr, flush=True)
        return False


def update_keep_alive(node_ip, miner_type):
    timestamp = int(time.time())
    redis.hset("mining_pool", node_ip, mapping={
        "last_keep_alive": timestamp,
        "miner_type": miner_type
    })


def check_node_status(node_ip):
    try:
        # Función para verificar el estado de un nodo
        current_time = int(time.time())
        last_keep_alive = redis.hget("mining_pool", node_ip)
        if last_keep_alive:
            last_keep_alive = int(last_keep_alive.decode())
            if current_time - last_keep_alive <= EXPIRATION_TIME:
                return True
        return False
    except redis_exceptions.RedisError as error:
        print(f"Redis error: {error}", file=sys.stderr, flush=True)
        return False


def get_gpu_active_nodes():
    # Función para obtener la cantidad de nodos activos
    try:
        gpu_active_nodes = 0
        all_nodes = redis.hkeys("mining_pool")
        for node_ip in all_nodes:
            if check_node_status(node_ip.decode()):
                active_nodes += 1
        return gpu_active_nodes
    except redis_exceptions.RedisError as error:
        print(f"Redis error: {error}", file=sys.stderr, flush=True)
        return 0


def check_pool_status():
    gpu_active_nodes = get_gpu_active_nodes()
    if (gpu_active_nodes == 0):
        print("Creating cloud miners...")
        # TODO:  Iniciar mineros CPU en la nube


def check_queue_status():
    print("Checking queue status...")
    # TODO: Iniciar mineros CPU en la nube


start_cronjob(check_pool_status, EXPIRATION_TIME)


@ app.route("/status")
def status():
    return jsonify({
        "status": "200",
        "description": "Coordinator proccess is executing..."
    })


@ app.route("/keep-alive", methods=['POST'])
def status():
    miner_information = ast.literal_eval(
        request.get_data().decode("utf-8"))  # {address, miner_type}
    update_keep_alive(
        miner_information["address"], miner_information["miner_type"])

    return jsonify({
        "status": "200",
        "description": "Pool status updated successfully"
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
