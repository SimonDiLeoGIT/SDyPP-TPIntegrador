import redis
import os


def redis_connect():
    redis_port = os.environ.get('REDIS_PORT')
    redis_host = os.environ.get('REDIS_HOST')

    return redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
