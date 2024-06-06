import json


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
