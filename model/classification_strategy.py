from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from .email import Email

class ClassificationStrategy(ABC):
    @abstractmethod
    def classify(self, email: Email) -> str:
        """
        Classify the email using the specific strategy
        """
        pass

class RandomForestStrategy(ClassificationStrategy):
    def __init__(self, n_estimators=1000, random_state=0):
        from sklearn.ensemble import RandomForestClassifier
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced_subsample'
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.classifier.fit(X_train, y_train)
    
    def classify(self, email: Email) -> str:
        features = email.to_features()
        prediction = self.classifier.predict(features.reshape(1, -1))
        return str(prediction[0])

class AdaBoostStrategy(ClassificationStrategy):
    def __init__(self, n_estimators=50, random_state=0):
        from sklearn.ensemble import AdaBoostClassifier
        self.classifier = AdaBoostClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.classifier.fit(X_train, y_train)
    
    def classify(self, email: Email) -> str:
        features = email.to_features()
        prediction = self.classifier.predict(features.reshape(1, -1))
        return str(prediction[0])

class ClassifierContext:
    def __init__(self, strategy: ClassificationStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: ClassificationStrategy):
        self._strategy = strategy
    
    def train_classifier(self, X_train: np.ndarray, y_train: np.ndarray):
        if hasattr(self._strategy, 'train'):
            self._strategy.train(X_train, y_train)
    
    def classify_email(self, email: Email) -> str:
        return self._strategy.classify(email)