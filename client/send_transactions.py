import time
import requests
import hashlib
import random


def get_md5():
    # Genera un string aleatorio de longitud 16 para usar como base para el hash
    random_string = ''.join(random.choice(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for i in range(16))
    md5_hash = hashlib.md5(random_string.encode()).hexdigest()
    return md5_hash


def send_transaction():
    sender = get_md5()
    receiver = get_md5()
    amount = random.randint(1, 1000)
    data = {
        "sender": sender,
        "receiver": receiver,
        "amount": amount
    }
    # Reemplaza con la URL a la que deseas enviar la petición POST
    url = 'http://coordinator.inventomate.xyz:5000/transaction'

    try:
        response = requests.post(url, json=data)
        print(
            f'Petición POST enviada a {url}. Respuesta: {response.status_code}')
    except requests.exceptions.RequestException as e:
        print(f'Error al enviar la petición POST: {e}')


if __name__ == '__main__':
    while True:
        for i in range(3):
            send_transaction()
        time.sleep(3*75)
