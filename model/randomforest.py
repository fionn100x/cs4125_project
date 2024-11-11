# This file is not used anymore as the code has been moved to
# classification_strategy.py and modelling.py


import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from numpy import *
import random
from model.base import BaseModel
from model.classification_strategy import RandomForestStrategy, ClassifierContext

num_folds = 0
seed =0
# Data
np.random.seed(seed)
random.seed(seed)

# This file already contain the code for implementing randomforest model
# Carefully observe the methods below and try calling them in modelling.py

class RandomForest(BaseModel):
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.strategy = RandomForestStrategy()
        self.context = ClassifierContext(self.strategy)
        self.predictions = None

    def train(self, data) -> None:
        self.context.train_classifier(data.X_train, data.y_train)
    
    def predict(self, X_test) -> np.ndarray:
        self.predictions = self.context.classify(X_test)
        return self.predictions

    def data_transform(self) -> None:
        pass
