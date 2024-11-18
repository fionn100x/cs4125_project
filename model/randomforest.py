import numpy as np
import pandas as pd

from observerPattern.subject import Subject
from .base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from observerPattern.model_subject import ModelSubject

class RandomForest(BaseModel, ModelSubject):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        # Initialize BaseModel
        BaseModel.__init__(self)
        # Initialize ModelSubject
        ModelSubject.__init__(self)
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = RandomForestClassifier(n_estimators=1000, random_state=0, class_weight='balanced_subsample')
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        self.notify("Random Forrest Training Started")
        self.mdl = self.mdl.fit(data.get_X_train(), data.get_type_y_train())
        self.notify("Training Completed")

    def predict(self, X_test: pd.Series):
        self.notify("Random Forrest Prediction Started")
        self.predictions = self.mdl.predict(X_test)
        self.notify("Prediction Completed")

    def print_results(self, data):
        self.notify("Random Forrest Evaluation Started")
        print(classification_report(data.get_type_y_test(), self.predictions))
        self.notify("Random Forrest Evaluation Completed")
    def data_transform(self) -> None:
        # Ensure embeddings and labels are properly aligned
        self.embeddings, self.labels = self.embeddings, self.y
