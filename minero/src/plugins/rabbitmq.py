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

            # Declaro exchange para las tasks de mineros
            channel.exchange_declare(
                exchange='workers', exchange_type='fanout', durable=True)

            # Declara una cola temporal exclusiva
            result = channel.queue_declare(queue='', exclusive=True)
            queue_name = result.method.queue

            # Vincula la cola al intercambio fanout
            channel.queue_bind(exchange='workers', queue=queue_name)

            # return channel
            return channel, queue_name
        except pika.exceptions.AMQPConnectionError:
            return None

    while True:
        channel = connect()
        if channel:
            break
        print("Failed to connect to RabbitMQ. Retrying in 5 seconds...")
        time.sleep(5)

    return channel
