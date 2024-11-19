import numpy as np
import pandas as pd
from .base import BaseModel
from observerPattern.model_subject import ModelSubject
from .email import Email
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

class RandomForest(BaseModel, ModelSubject):
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray, vectorizer) -> None:
        super().__init__()
        ModelSubject.__init__(self)
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.vectorizer = vectorizer
        self.model = RandomForestClassifier()
        self.predictions = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.notify("RandomForest Training Started")
        self.model.fit(X, y)
        self.notify("RandomForest Training Completed")

    def predict(self, content: str):
        self.notify("RandomForest Predicting Email")
        email = Email(content=content, summary="")
        features = email.to_features(self.vectorizer)
        pred = self.model.predict([features])
        self.notify("RandomForest Email Predicted")
        return pred

    def print_results(self, data):
        self.notify("RandomForest Printing results")
        predictions = self.model.predict(data.get_X_test())
        print(classification_report(data.get_type_y_test(), predictions))
        self.notify("RandomForest Results Printed")

    def data_transform(self) -> None:
        self.embeddings, self.labels = self.embeddings, self.y