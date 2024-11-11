from abc import ABC, abstractmethod
import numpy as np
from email import Email

class EmailClassifier(ABC):
    @abstractmethod
    def classify(self, email: Email) -> str:
        """
        """
        pass

class BaseClassifier(EmailClassifier):
    def __init__(self):
        self.model = None  #

    def classify(self, email: Email) -> str:
        """

        :param email:
        :return:
        """
        features = email.to_features()
        prediction = self.model.predict(features.reshape(1, -1))
        return str(prediction[0])