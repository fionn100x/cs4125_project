import numpy as np
from .base import BaseModel
from .classification_strategy import HistGradientBoostingStrategy, ClassifierContext
from observerPattern.model_subject import ModelSubject
from .email import Email
from sklearn.metrics import classification_report 
from sklearn.ensemble import HistGradientBoostingClassifier

class HistGradientBoosting(BaseModel, ModelSubject):
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray, vectorizer) -> None:
        super().__init__()
        ModelSubject.__init__(self)
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.vectorizer = vectorizer
        self.model = HistGradientBoostingClassifier()
        self.predictions = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.notify("HistGB Training Started")
        self.model.fit(X, y)
        self.notify("Training Completed")

    def predict(self, content: str):
        self.notify("HistGB Prediction Started")
        email = Email(content=content, summary="")
        features = email.to_features(self.vectorizer)
        pred = self.model.predict([features])
        self.notify("Prediction Completed")
        return pred

    def print_results(self, data):
        self.notify("HistGB Evaluation Started")
        predictions = self.model.predict(data.get_X_test())
        print(classification_report(data.get_type_y_test(), predictions))
        self.notify("HistGB Evaluation Completed")

    def data_transform(self) -> None:
        self.embeddings, self.labels = self.embeddings, self.y