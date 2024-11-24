from factoryPattern.classifier_factory import ClassifierFactory
from typing import List
from model.classification_strategy import ClassifierContext, AdaBoostStrategy, RandomForestStrategy, \
    HistGradientBoostingStrategy, SGDStrategy, VotingStrategy
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

def model_predict(data, df, name: str):
    strategies = {
                'adaboost': AdaBoostStrategy(),
                'randomforest': RandomForestStrategy(),
                'histgb': HistGradientBoostingStrategy(),
                'sgd':SGDStrategy(),
                'voting': VotingStrategy()
    }

    strategy = strategies.get(name.lower())
    if not strategy:
        raise ValueError(f"Unknown model: {name}. Available models: {list(strategies.keys())}")

    classifier = ClassifierContext(strategy)

    # Initialize and train the model
    classifier.train_classifier(data.get_X_train(), data.get_type_y_train(), data.vectorizer)

    # Create email objects from test data
    predictions = []
    for _, row in data.df_test.iterrows():
        content = row['Interaction content']
        summary = row['Ticket Summary']
        email = Email(content=str(content), summary=str(summary))
        pred = classifier.classify_email(email)
        predictions.append(pred[0])  # Get the prediction from the array

    # Print results
    strategy.model.print_results(data)

    return predictions