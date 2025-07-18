from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import List, Dict
import numpy as np
import pickle

from sklearn.naive_bayes import MultinomialNB



class Model:
    def __init__(self, pretrained_path=None):
        self.model = MultinomialNB()

        if pretrained_path:  
            loaded = self.load(pretrained_path)

            if isinstance(loaded, MultinomialNB):
                self.model = loaded
            elif isinstance(loaded, Model):
                self.model = loaded.model



    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)


    def evaluate(self, x: List[List[float]], y: List[int]) -> Dict:
        prediction = self.predict(x)

        prediction = np.argmax(prediction, axis=1)

        if len(y) > 0 and isinstance(y[0], (list, np.ndarray)):
            y = np.argmax(y, axis=1)

        return {
            "accuracy": accuracy_score(y, prediction),
            "f1": f1_score(y, prediction, average="weighted"),
            "roc_auc": roc_auc_score(y, prediction, average="weighted")
        }


    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)
