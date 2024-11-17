#This is a main file: The controller. All methods will directly on directly be called here
import os
import joblib
import numpy as np
from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
from config_manager import ConfigurationManager
from model.randomforest import RandomForest
from command import Command, ClassifyEmailCommand, MarkAsSpamCommand

seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    #load the input data
    df = get_input_data()
    return  df

def preprocess_data(df):
    # De-duplicate input data
    df =  de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)

class ClassifierManager:
    """
    Singleton class for managing classifier operations.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClassifierManager, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        print("Initializing ClassifierManager...")

    def train_random_forest_model(self, data: Data):
        labels = data.get_type_y_train()
        rf_model = RandomForest('RandomForest', data.get_X_train(), labels)
        rf_model.train(data)
        os.makedirs('models', exist_ok=True)
        joblib.dump(rf_model, 'models/random_forest_classifier.pkl')
        print("RandomForest model trained and saved as 'models/random_forest_classifier.pkl'.")

    def classify_emails(self, data: Data):
        try:
            rf_model = joblib.load('models/random_forest_classifier.pkl')
            print("RandomForest model loaded successfully.")
            rf_model.predict(data.get_X_test())
            data.test_df['Predicted Category'] = rf_model.predictions
            os.makedirs('results', exist_ok=True)
            result_path = 'results/result.csv'
            data.test_df.to_csv(result_path, index=False)
            print(f"Classification completed and results saved to '{result_path}'.")
            print("Here are the first few results:")
            print(data.test_df[['Ticket id', 'Predicted Category']].head())
        except FileNotFoundError:
            print("No pre-trained RandomForest model found. Please train the model first.")


# Code will start executing from following line
if __name__ == '__main__':
    
    # pre-processing steps
    df = load_data()
    if not df.empty:
        df = preprocess_data(df)
        df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
        df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    
    # data transformation
    X, group_df = get_embeddings(df)
    # data modelling
    data = get_data_object(X, df)
    classifier_manager = ClassifierManager()
        classifier_manager.train_random_forest_model(data)
        
        # Create and execute commands based on user or system actions
        commands = [
            ClassifyEmailCommand(classifier_manager, data),
            MarkAsSpamCommand(data)
        ]

        for command in commands:
            command.execute()
    else:
        print("No data loaded. Please check your file path and data integrity.")

    # modelling
    perform_modelling(data, df, 'name')

