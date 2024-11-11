# sgd_classifier.py (in model directory)

from sklearn.linear_model import SGDClassifier
import numpy as np
from .base_classifier import BaseClassifier
from .email import Email

class SGDClassifier(BaseClassifier):
    def __init__(self, loss='hinge', penalty='l2', max_iter=1000, random_state=0):
        super().__init__()
        self.model = SGDClassifier(loss=loss, penalty=penalty, max_iter=max_iter, random_state=random_state)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict(self, features: np.ndarray) -> int:
        return self.model.predict(features.reshape(1, -1))[0]