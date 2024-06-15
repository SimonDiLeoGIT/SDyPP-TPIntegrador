import json
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
            'data': self.data,
            'hash': self.hash,
            "nonce": self.nonce,
        }

    def get_block_content_hash(self):
        serialized_data = [json.dumps(obj) for obj in self.data]
        data_as_string = ''.join(serialized_data)

        block_content = f"{str(self.index).strip()}{str(self.timestamp).strip()}{data_as_string}{str(self.previous_hash).strip()}"

        block_content_hash = md5(block_content.encode("utf-8")).hexdigest()

        return block_content_hash
