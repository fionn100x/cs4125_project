from commandPattern.command import Command
from model.email import Email
from model.classification_strategy import ClassifierContext

class ClassifyEmailCommand(Command):
    def __init__(self, classifier_context: ClassifierContext, email: Email):
        self.classifier_context = classifier_context
        self.email = email
        self.prediction = None

    def execute(self) -> None:
        print(f"Executing classification for email: {self.email.content}")
        self.prediction = self.classifier_context.classify_email(self.email)
        print(f"Prediction: {self.prediction}")

    def undo(self) -> None:
        self.prediction = None