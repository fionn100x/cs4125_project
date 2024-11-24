import os
import random
import numpy as np
import pandas as pd

from factoryPattern.preprocessor_factory import PreprocessorFactory
from preprocess import *
from embeddings import *
from modelling.data_model import Data
from model.email import Email
from modelling.modelling import ModellingManager
from model.classification_strategy import (
    ClassifierContext,
    RandomForestStrategy,
    AdaBoostStrategy,
    HistGradientBoostingStrategy,
    VotingStrategy  
)
from commandPattern.command import Command
from commandPattern.commands import SetStrategyCommand, TrainModelCommand, PredictCommand
from commandPattern.classify_email_command import ClassifyEmailCommand
from commandPattern.invoker import CommandInvoker
from preprocessing_pipeline import PreprocessingPipeline

# Set random seed for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)

class ClassifierManager:
    """
    Singleton class for managing classifier operations.
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClassifierManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not ClassifierManager._initialized:
            self.classifier_context = None
            self.modelling_manager = ModellingManager()  # Use ModellingManager singleton
            ClassifierManager._initialized = True
            print("Initializing ClassifierManager...")

    def set_strategy(self, model_name: str):
        """Set the classification strategy based on model name."""
        self.classifier_context = self.modelling_manager.create_classifier_context(model_name)
        print(f"Strategy set to {model_name}")

    def train_model(self, data: Data):
        """Train the selected model using the provided data."""
        if self.classifier_context is None:
            raise ValueError("No classification strategy selected. Call set_strategy first.")
        
        self.classifier_context.train_classifier(
            data.get_X_train(),
            data.get_type_y_train(),
            data.vectorizer
        )

    def predict(self, data: Data):
        """Make predictions using the trained model."""
        if self.classifier_context is None:
            raise ValueError("No classification strategy selected. Call set_strategy first.")
        
        # Use ModellingManager to process test data
        emails = self.modelling_manager.process_test_data(data)
        predictions = []
        invoker = CommandInvoker()

        for email in emails:
            command = ClassifyEmailCommand(self.classifier_context, email)
            invoker.add_command(command)
        
        invoker.execute_commands()

        for command in invoker._history:
            predictions.append(command.prediction[0])
        
        test_df = data.get_test_df()
        test_df['Predicted Category'] = predictions
        
        # Save results
        os.makedirs('results', exist_ok=True)
        result_path = 'results/result.csv'
        test_df.to_csv(result_path, index=False)
        
        # Evaluate model
        true_labels = data.get_type_y_test()
        self.modelling_manager.evaluate_model(true_labels, predictions, 
                                            self.classifier_context.strategy.__class__.__name__)
        
        print(f"Classification completed and results saved to '{result_path}'")
        return predictions

def load_data():
    """Load the dataset."""
    df = pd.read_csv("data/AppGallery.csv")
    return df

def preprocess_data(df):
    """Preprocess the loaded data dynamically using a pipeline."""
    # Define the desired preprocessing steps
    preprocessing_steps = [
        PreprocessorFactory.create_preprocessor("deduplication"),
        PreprocessorFactory.create_preprocessor("noise_removal"),
        PreprocessorFactory.create_preprocessor("translation")
    ]

    # Create and apply the preprocessing pipeline
    pipeline = PreprocessingPipeline(preprocessing_steps)
    processed_df = pipeline.process(df)
    processed_df['y2'] = processed_df['Type 2']  # Add any additional required columns
    return processed_df

def get_embeddings(df):
    """Get TF-IDF embeddings for the data."""
    X, vectorizer = get_tfidf_embd(df)
    return X, vectorizer, df

def get_data_object(X, vectorizer, df):
    """Create a Data object with the processed data."""
    return Data(X, vectorizer, df)

def perform_modelling(data: Data, df: pd.DataFrame, model_name: str):
    """Perform the modeling process using the singleton ClassifierManager."""
    try:
        classifier_manager = ClassifierManager()
        invoker = CommandInvoker()

        # Create and add commands to the invoker
        set_strategy_command = SetStrategyCommand(classifier_manager, model_name)
        train_model_command = TrainModelCommand(classifier_manager, data)
        predict_command = PredictCommand(classifier_manager, data)

        invoker.add_command(set_strategy_command)
        invoker.add_command(train_model_command)
        invoker.add_command(predict_command)

        # Execute all commands
        invoker.execute_commands()

        # Get predictions and print results
        predictions = classifier_manager.predict(data)
        print("Predictions:", predictions)
    except Exception as e:
        print(f"Error during modeling: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        df = load_data()
        df = preprocess_data(df)
        df['Interaction content'] = df['Interaction content'].astype(str)
        df['Ticket Summary'] = df['Ticket Summary'].astype(str)

        available_models = ['adaboost', 'randomforest', 'histgb', 'voting', 'sgd']
        model_name = input(f"Select model ({'/'.join(available_models)}): ").lower()

        X, vectorizer, group_df = get_embeddings(df)
        data = get_data_object(X, vectorizer, df)
        perform_modelling(data, df, model_name)
    except Exception as e:
        print(f"An error occurred: {str(e)}")