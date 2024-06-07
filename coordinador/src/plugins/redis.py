import redis
import os
import time
import redis.exceptions


def redis_connect():
    redis_port = os.environ.get('REDIS_PORT')
    redis_host = os.environ.get('REDIS_HOST')

    def connect():
        try:
            connection = redis.Redis(
                host=redis_host, port=redis_port, decode_responses=True)

            return connection
        except redis.exceptions.ConnectionError:
            return None

    while True:
        connection = connect()
        if connection:
            break
        print("Failed to connect to Redis. Retrying in 5 seconds...")
        time.sleep(5)

    return connection
