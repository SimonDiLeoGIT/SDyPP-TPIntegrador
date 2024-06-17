import pika
import os
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

            # Declaro queue para las transacciones
            channel.exchange_declare(
                exchange='blockchain', exchange_type='direct', durable=True, auto_delete=False)
            channel.queue_declare(queue='transactions', durable=True)
            channel.queue_bind(
                exchange='blockchain', queue='transactions', routing_key='tx')

            # Declaro exchange para las tasks de mineria
            channel.queue_declare(queue='blocks', durable=True)
            channel.queue_bind(
                exchange='blockchain', queue='blocks', routing_key='bl')

            return channel
        except pika.exceptions.AMQPConnectionError:
            return None

    while True:
        channel = connect()
        if channel:
            break
        print("Failed to connect to RabbitMQ. Retrying in 5 seconds...")
        time.sleep(5)

    return channel
