import json
import sys
import os
from hashlib import md5


class Block:
    def __init__(self,  data, timestamp, hash, previous_hash, nonce, index):
        self.timestamp = timestamp
        self.hash = hash
        self.previous_hash = previous_hash
        self.data = json.dumps(data)
        self.index = index
        self.nonce = nonce

    def to_dict(self):
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'data': self.data,
            'hash': self.hash,
            "nonce": self.nonce,
        }

        # Recalcular el hash md5(nonce + md5(index+timestamp+data+previous_hash)) que calcularon los mineros para ver si es válido
    def validate(self):
        hash_challengue = os.environ.get("HASH_CHALLENGUE")
        if (not self.hash.startswith(hash_challengue)):
            return False

        data_as_string = ''.join(
            [json.dumps(obj) for obj in self.data])

        block_content = data_as_string + str(self.index).strip() + \
            str(self.previous_hash).strip() + str(self.timestamp).strip()

        nonce_bytes = str(self.nonce).strip().encode("utf-8")
        block_content_bytes = block_content.encode("utf-8")

        recalculated_block_hash = md5(
            nonce_bytes + block_content_bytes).hexdigest()

        print(recalculated_block_hash, file=sys.stdout, flush=True)

        return recalculated_block_hash == self.hash
