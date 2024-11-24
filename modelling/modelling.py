from factoryPattern.classifier_factory import ClassifierFactory
from typing import List
from model.classification_strategy import ClassifierContext
from model.email import Email
from sklearn.metrics import classification_report

class ModellingManager:
    """
    Handles the modeling operations and interfaces with ClassifierManager.
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModellingManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not ModellingManager._initialized:
            ModellingManager._initialized = True

    def get_available_models(self) -> List[str]:
        """Returns list of available model names from the factory."""
        return ClassifierFactory.get_supported_models()

    def create_classifier_context(self, name: str) -> ClassifierContext:
        """Create a classifier context with the specified strategy using the factory."""
        strategy = ClassifierFactory.create_classifier(name)  # Use the factory to create the strategy
        return ClassifierContext(strategy)

    def process_test_data(self, data) -> List[Email]:
        """Process test data into Email objects."""
        emails = []
        for _, row in data.df_test.iterrows():
            content = str(row['Interaction content'])
            summary = str(row['Ticket Summary'])
            emails.append(Email(content=content, summary=summary))
        return emails

    def evaluate_model(self, true_labels: List, predictions: List, model_name: str) -> None:
        """Print model evaluation metrics."""
        print(f"\nModel Evaluation Results for {model_name}")
        print("-" * 50)
        print(classification_report(true_labels, predictions))
