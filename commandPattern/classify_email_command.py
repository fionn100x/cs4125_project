from commandPattern.command import Command
from model.email import Email
from model.classification_strategy import ClassifierContext

class ClassifyEmailCommand(Command):
    def __init__(self, classifier_context: ClassifierContext, email: Email):
        self.classifier_context = classifier_context
        self.email = email
        self.prediction = None

    def execute(self) -> None:
        self.prediction = self.classifier_context.classify_email(self.email)

    def undo(self) -> None:
        self.prediction = None