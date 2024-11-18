import numpy as np
from .base import BaseModel
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

class HistGradientBoosting(BaseModel):
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = HistGradientBoostingClassifier(random_state=0)
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        """Train the Histogram-based Gradient Boosting model."""
        self.mdl.fit(data.get_X_train(), data.get_type_y_train())

    def predict(self, X_test: np.ndarray):
        """Predict labels for the given test data."""
        self.predictions = self.mdl.predict(X_test)

    def print_results(self, data) -> None:
        """Print classification report."""
        print(classification_report(data.get_type_y_test(), self.predictions))

    def data_transform(self) -> None:
        """Prepare the data for training and prediction."""
        self.embeddings, self.labels = self.embeddings, self.y
