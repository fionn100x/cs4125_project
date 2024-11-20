from typing import List, Dict, Any
from sklearn.metrics import classification_report
from model.classification_strategy import (
    AdaBoostStrategy,
    RandomForestStrategy,
    HistGradientBoostingStrategy,
    ClassifierContext
)
from model.email import Email

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
            self._strategies: Dict[str, Any] = {
                'adaboost': AdaBoostStrategy(),
                'randomforest': RandomForestStrategy(),
                'histgb': HistGradientBoostingStrategy()
            }
            ModellingManager._initialized = True

    def get_available_models(self) -> List[str]:
        """Returns list of available model names."""
        return list(self._strategies.keys())

    def get_strategy(self, name: str):
        """Get the strategy for the specified model name."""
        strategy = self._strategies.get(name.lower())
        if not strategy:
            raise ValueError(
                f"Unknown model: {name}. Available models: {self.get_available_models()}"
            )
        return strategy

    def create_classifier_context(self, name: str) -> ClassifierContext:
        """Create a classifier context with the specified strategy."""
        strategy = self.get_strategy(name)
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
    # Get the singleton instance
    modelling_manager = ModellingManager()
    
    try:
        # Create classifier context
        classifier = modelling_manager.create_classifier_context(name)
        
        # Train the model
        classifier.train_classifier(
            data.get_X_train(),
            data.get_type_y_train(),
            data.vectorizer
        )
        
        # Process test data and make predictions
        emails = modelling_manager.process_test_data(data)
        predictions = [classifier.classify_email(email)[0] for email in emails]
        
        # Print results
        classifier.strategy.print_results(data)
        
        # Evaluate model
        true_labels = data.get_type_y_test()
        modelling_manager.evaluate_model(true_labels, predictions, name)
        
        return predictions
        
    except Exception as e:
        print(f"Error during model prediction: {str(e)}")
        raise