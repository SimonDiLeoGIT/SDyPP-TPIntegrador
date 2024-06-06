import json


class Block:
    def __init__(self,  data, timestamp, hash, previous_hash):
        self.timestamp = timestamp
        self.hash = hash
        self.previous_hash = previous_hash
        self.data = json.dumps(data)

    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'hash': self.hash,
            'previous_hash': self.previous_hash,
            'data': self.data,
        }
