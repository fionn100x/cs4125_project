import numpy as np
from .base import BaseModel
from observerPattern.model_subject import ModelSubject
from .email import Email
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

class LogisticRegressionModel(BaseModel, ModelSubject):
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray, vectorizer, **kwargs) -> None:
        super().__init__()
        ModelSubject.__init__(self)
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.vectorizer = vectorizer
        self.model = LogisticRegression(**kwargs)
        self.predictions = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.notify("Logistic Regression Training Started")
        self.model.fit(X, y)
        self.notify("Logistic Regression  Training Completed")

    def predict(self, content: str):
        self.notify("Logistic Regression  Predicting Email")
        email = Email(content=content, summary="")
        features = email.to_features(self.vectorizer)
        pred = self.model.predict([features])
        self.notify("Logistic Regression  Email Predicted")
        return pred

    def print_results(self, data):
        self.notify("Logistic Regression Printing Results")
        predictions = self.model.predict(data.get_X_test())
        print(classification_report(data.get_type_y_test(), predictions))
        self.notify("Logistic Regression Results Printed")

    def data_transform(self) -> None:
        self.embeddings, self.labels = self.embeddings, self.y
