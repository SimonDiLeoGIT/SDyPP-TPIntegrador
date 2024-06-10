import hashlib


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
    for nonce in range(start_nonce, end_nonce + 1):
        test_string = f"{nonce}{base_string}"
        hash_result = hashlib.md5(test_string.encode()).hexdigest()
        if hash_result.startswith(target_hash_prefix):
            return nonce, hash_result
    return None


def main(target_hash_prefix, base_string, start_nonce, end_nonce):
    result = find_nonce_with_prefix(
        target_hash_prefix, base_string, start_nonce, end_nonce)

    if result:
        nonce, hash_result = result
        return f"Nonce encontrado: {nonce}\nHash correspondiente: {hash_result}"
    else:
        return "No se encontró ningún nonce que cumpla con la condición en el rango proporcionado."


# Ejemplo de uso
if __name__ == "__main__":
    target_hash_prefix = "0000"
    base_string = "example"
    start_nonce = 0
    end_nonce = 1000000

    result = main(target_hash_prefix, base_string, start_nonce, end_nonce)
    print(result)
