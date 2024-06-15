import json
import sys
import os
from hashlib import md5


class Block:
    def __init__(self,  data, timestamp, hash, previous_hash, nonce, index):
        self.timestamp = timestamp
        self.hash = hash
        self.previous_hash = previous_hash
        self.data = data
        self.index = index
        self.nonce = nonce

    def to_dict(self):
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'data': json.dumps(self.data),
            'hash': self.hash,
            "nonce": self.nonce,
        }

    # Recalcular el hash md5(nonce + md5(index+timestamp+data+previous_hash)) que calcularon los mineros para ver si es válido
    def validate(self):
        hash_challenge = os.environ.get("HASH_CHALLENGE")
        if (not self.hash.startswith(hash_challenge)):
            return False

        serialized_data = [json.dumps(obj) for obj in self.data]
        data_as_string = ''.join(serialized_data)

        block_content = f"{str(self.index).strip()}{str(self.timestamp).strip()}{data_as_string}{str(self.previous_hash).strip()}"

        # Calcula el hash del contenido del bloque
        block_content_hash = md5(block_content.encode("utf-8"))

        # Calcula el hash del bloque agregandole el hash antes
        md5_input = f"{int(self.nonce)}{block_content_hash}"
        recalculated_block_hash = md5(md5_input.encode("utf-8")).hexdigest()

        # Valida si el hash calculado por el minero (self.hash) es igual al hash calculado
        return recalculated_block_hash == self.hash
