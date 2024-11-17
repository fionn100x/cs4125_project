import numpy as np
import pandas as pd
from .base import BaseModel
from observerPattern.model_subject import ModelSubject
from .email import Email
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier as SklearnSGDClassifier

class SGDClassifier(BaseModel, ModelSubject):
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray, vectorizer, loss='hinge', penalty='l2', max_iter=1000, random_state=0) -> None:
        super().__init__()
        ModelSubject.__init__(self)
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.vectorizer = vectorizer
        self.model = SklearnSGDClassifier(loss=loss, penalty=penalty, max_iter=max_iter, random_state=random_state)
        self.predictions = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.notify("SGD Training Started")
        self.model.fit(X, y)
        self.notify("SGD Training Completed")

    def predict(self, content: str):
        self.notify("SGD Predicting Email")
        email = Email(content=content, summary="")
        features = email.to_features(self.vectorizer)
        pred = self.model.predict([features])
        self.notify("SGD Email Classified")
        return pred

    def print_results(self, data):
        self.notify("SGD Printing Results")
        predictions = self.model.predict(data.get_X_test())
        print(classification_report(data.get_type_y_test(), predictions))
        self.notify("SGD Results Printed")

    def data_transform(self) -> None:
        self.embeddings, self.labels = self.embeddings, self.y