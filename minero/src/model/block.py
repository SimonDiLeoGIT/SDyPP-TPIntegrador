import json


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

    def get_block_content_as_string(self):
        serialized_objects = [json.dumps(obj) for obj in self.data]

        data_as_string = '[' + ','.join(serialized_objects) + ']'

        block_content = f"{data_as_string}{str(self.index).strip()}{str(self.previous_hash).strip()}{str(self.timestamp).strip()}"

        return block_content
