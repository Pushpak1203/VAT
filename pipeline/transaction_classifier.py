"""Step 2: transaction classification."""
from models.deberta_classifier import DebertaClassifier


class TransactionClassifier:
    def __init__(self, settings):
        self.model = DebertaClassifier(settings)

    def classify(self, transaction):
        return self.model.classify(transaction)
