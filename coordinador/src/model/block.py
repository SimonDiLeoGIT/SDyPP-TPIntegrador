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

        serialized_objects = [json.dumps(obj) for obj in self.data]
        # Une las cadenas JSON con comas y encierra en corchetes para formar una lista JSON válida
        data_as_string = '[' + ','.join(serialized_objects) + ']'

        block_content = f"{data_as_string}{str(self.index).strip()}{str(self.previous_hash).strip()}{str(self.timestamp).strip()}"

        nonce_bytes = str(self.nonce).strip().encode("utf-8")
        block_content_bytes = block_content.encode("utf-8")

        recalculated_block_hash = md5(
            nonce_bytes + block_content_bytes).hexdigest()

        print(recalculated_block_hash, file=sys.stdout, flush=True)

        return recalculated_block_hash == self.hash
