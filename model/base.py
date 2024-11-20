from abc import ABC, abstractmethod
import utils
import pandas as pd
import numpy as np

class BaseModel(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        """
        Train the model using ML Models for Multi-class and multi-label classification.
        """
        pass

    @abstractmethod
    def predict(self) -> int:
        """
        Predict labels for given data.
        """
        pass

    @abstractmethod
    def data_transform(self) -> None:
        """
        Perform any required data transformation.
        """
        pass

    def build(self, values={}):
        values = values if isinstance(values, dict) else utils.string2any(values)
        self.__dict__.update(self.defaults)
        self.__dict__.update(values)
        return self
