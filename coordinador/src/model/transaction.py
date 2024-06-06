from flask import jsonify


class Transaction:
    def __init__(self, id, sender, receiver, amount, signature, timestamp):
        self.id = id
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.signature = signature
        self.timestamp = timestamp

    def to_dict(self):
        return {
            'id': self.id,
            'sender': self.sender,
            'receiver': self.receiver,
            'amount': self.amount,
            'signature': self.signature,
            'timestamp': self.timestamp
        }
