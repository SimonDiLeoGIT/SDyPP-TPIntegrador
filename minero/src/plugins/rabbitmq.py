import pika
import os
import sys
import time


def rabbit_connect():
    rabbit_user = os.environ.get("RABBITMQ_USER")
    rabbit_password = os.environ.get("RABBITMQ_PASSWORD")
    rabbit_host = os.environ.get("RABBITMQ_HOST")

    def connect():
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=rabbit_host,
                    credentials=pika.PlainCredentials(
                        rabbit_user, rabbit_password),
                    heartbeat=3600
                )
            )
            channel = connection.channel()

            # Declaro exchange para las tasks de mineros
            channel.exchange_declare(
                exchange='blockchain', exchange_type='direct', durable=True, auto_delete=False)
            channel.queue_declare(queue='workers', durable=True)
            channel.queue_bind(
                exchange='blockchain', queue='workers', routing_key='w')

            # return channel
            return channel
        except pika.exceptions.AMQPConnectionError:
            return None

    while True:
        channel = connect()
        if channel:
            break
        print("Failed to connect to RabbitMQ. Retrying in 5 seconds...",
              file=sys.stdout, flush=True)
        time.sleep(5)

    return channel
