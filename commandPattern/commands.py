from commandPattern.command import Command
from model.classification_strategy import ClassifierContext
from modelling.data_model import Data

class SetStrategyCommand(Command):
    def __init__(self, classifier_manager, model_name: str):
        self.classifier_manager = classifier_manager
        self.model_name = model_name

    def execute(self) -> None:
        self.classifier_manager.set_strategy(self.model_name)

    def undo(self) -> None:
        # Undo logic for setting strategy (if applicable)
        pass

class TrainModelCommand(Command):
    def __init__(self, classifier_manager, data: Data):
        self.classifier_manager = classifier_manager
        self.data = data

    def execute(self) -> None:
        self.classifier_manager.train_model(self.data)

    def undo(self) -> None:
        # Undo logic for training model (if applicable)
        pass

class PredictCommand(Command):
    def __init__(self, classifier_manager, data: Data):
        self.classifier_manager = classifier_manager
        self.data = data

    def execute(self) -> None:
        self.classifier_manager.predict(self.data)

    def undo(self) -> None:
        # Undo logic for prediction (if applicable)
        pass