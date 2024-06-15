from flask import jsonify


class Transaction:
    def __init__(self, id, sender, receiver, amount, timestamp):
        self.id = id
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.timestamp = timestamp

    def to_dict(self):
        return {
            'id': self.id,
            'sender': self.sender,
            'receiver': self.receiver,
            'amount': self.amount,
            'timestamp': self.timestamp
        }
