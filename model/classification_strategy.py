from abc import ABC, abstractmethod
import numpy as np
from typing import Any

from observerPattern.logging_observer import LoggingObserver
from .email import Email

class ClassificationStrategy(ABC):
    def __init__(self):
        self._trained = False
        self.model = None

    def initialize_model(self, embeddings: np.ndarray, y: np.ndarray, vectorizer) -> None:
        if self.model is None:
            self._initialize_model(embeddings, y, vectorizer)

    @abstractmethod
    def _initialize_model(self, embeddings: np.ndarray, y: np.ndarray, vectorizer) -> None:
        pass

    def train_classifier(self, X: np.ndarray, y: np.ndarray, vectorizer) -> None:
        if self.model is None:
            self.initialize_model(X, y, vectorizer)
        self.model.train(X, y)
        self._trained = True

    def print_results(self, data):
        self.model.print_results(data)

    def classify_email(self, email: Email) -> Any:
        if not self._trained:
            raise ValueError("Model must be trained before classification")
        return self.model.predict(email.content)

class AdaBoostStrategy(ClassificationStrategy):
    def _initialize_model(self, embeddings: np.ndarray, y: np.ndarray, vectorizer) -> None:
        from .adaboosting import AdaBoosting
        self.model = AdaBoosting(model_name="AdaBoost", embeddings=embeddings, y=y, vectorizer=vectorizer)
        # Initialize LoggingObserver and attach it to the model
        ada_logger = LoggingObserver(observer_name="AdaBoostLoggingObserver")
        self.model.attach(ada_logger)  # Attach the observer to the model
        self.model.data_transform()

class LogisticRegression(ClassificationStrategy):
    def _initialize_model(self, embeddings: np.ndarray, y: np.ndarray, vectorizer) -> None:
        from .logistic_regression import LogisticRegressionModel
        self.model = LogisticRegressionModel(model_name="LogisticRegression", embeddings=embeddings, y=y, vectorizer=vectorizer)
        # Initialize LoggingObserver and attach it to the model
        logistic_regression_logger = LoggingObserver(observer_name="LogisticRegressionObserver")
        self.model.attach(logistic_regression_logger)  # Attach the observer to the model
        self.model.data_transform()

class RandomForestStrategy(ClassificationStrategy):
    def _initialize_model(self, embeddings: np.ndarray, y: np.ndarray, vectorizer) -> None:
        from .randomforest import RandomForest
        self.model = RandomForest(model_name="RandomForest", embeddings=embeddings, y=y, vectorizer=vectorizer)
        # Initialize LoggingObserver and attach it to the model
        rf_logger = LoggingObserver(observer_name="RandomForestLoggingObserver")
        self.model.attach(rf_logger)  # Attach the observer to the model
        self.model.data_transform()

class HistGradientBoostingStrategy(ClassificationStrategy):
    def _initialize_model(self, embeddings: np.ndarray, y: np.ndarray, vectorizer) -> None:
        from .hist_gb import HistGradientBoosting
        self.model = HistGradientBoosting(
            model_name="HistGradientBoosting",
            embeddings=embeddings,
            y=y,
            vectorizer=vectorizer
        )
        # Attach a LoggingObserver for model notifications
        from observerPattern.logging_observer import LoggingObserver
        hist_logger = LoggingObserver(observer_name="HistGBLoggingObserver")
        self.model.attach(hist_logger)
        self.model.data_transform()

class VotingStrategy(ClassificationStrategy):
    def _initialize_model(self, embeddings: np.ndarray, y: np.ndarray, vectorizer) -> None:
        from .voting import VotingModel
        self.model = VotingModel(model_name="VotingModel", embeddings=embeddings, y=y, vectorizer=vectorizer)
        # Attach observer for logging
        voting_logger = LoggingObserver(observer_name="VotingLoggingObserver")
        self.model.attach(voting_logger)
        self.model.data_transform()

class SGDStrategy(ClassificationStrategy):
    def _initialize_model(self, embeddings: np.ndarray, y: np.ndarray, vectorizer) -> None:
        from .sgd_classifier import SGDClassifier
        self.model = SGDClassifier(model_name="SGDClassifier", embeddings=embeddings, y=y, vectorizer=vectorizer)
        sgd_logger = LoggingObserver(observer_name="SGDLoggingObserver")
        self.model.attach(sgd_logger)
        self.model.data_transform()

class ClassifierContext:
    def __init__(self, strategy: ClassificationStrategy):
        self.strategy = strategy

    def train_classifier(self, X: np.ndarray, y: np.ndarray, vectorizer) -> None:
        self.strategy.train_classifier(X, y, vectorizer)

    def print_results(self, data):
        self.strategy.print_results(data)

    def classify_email(self, email: Email) -> Any:
        return self.strategy.classify_email(email)
