import hashlib
import argparse
import time


def find_nonce_with_prefix(target_hash_prefix, base_string, start_nonce, end_nonce):
    """
    Encuentra un nonce tal que el hash MD5 del (nonce + base_string) comience con target_hash_prefix.
    Busca el nonce dentro del rango [start_nonce, end_nonce].

    Parameters:
    - target_hash_prefix: Prefijo del hash objetivo.
    - base_string: Cadena base a la que se concatenará el nonce.
    - start_nonce: Inicio del rango de nonces.
    - end_nonce: Fin del rango de nonces.

    Returns:
    - (nonce, hash): Nonce encontrado y el hash correspondiente.
    - None: Si no se encuentra ningún nonce que cumpla con la condición.
    """
    start_time = time.time()
    for nonce in range(start_nonce, end_nonce + 1):
        test_string = f"{nonce}{base_string}"
        hash_result = hashlib.md5(test_string.encode("utf-8")).hexdigest()
        if hash_result.startswith(target_hash_prefix):
            end_time = time.time()
            elapsed_time = end_time - start_time
            return nonce, hash_result, elapsed_time
    return None, None, None


def main(args):
    nonce, hash_result, elapsed_time = find_nonce_with_prefix(
        args.prefix, args.base_string, args.start_nonce, args.end_nonce)

    if nonce is not None:
        print(f"Nonce encontrado: {nonce}\nHash correspondiente: {hash_result}")
        print(f"Tiempo transcurrido: {elapsed_time:.6f} segundos")
    else:
        print("No se encontró ningún nonce que cumpla con la condición en el rango proporcionado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find nonce with prefix')
    parser.add_argument('prefix', type=str, help='Target hash prefix')
    parser.add_argument('base_string', type=str, help='Base string')
    parser.add_argument('start_nonce', type=int, help='Start nonce')
    parser.add_argument('end_nonce', type=int, help='End nonce')

    args = parser.parse_args()
    main(args)

# 