#This is a main file: The controller. All methods will directly on directly be called here
from turtle import pd

import np

from model.email import Email
from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
from model.base_classifier import BaseClassifier
from classifier_decorator import NoiseRemovalDecorator, TranslationDecorator
from model.sgd_classifier import SGDClassifier

import random
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
    base_classifier = SGDClassifier()
    classifier_with_decorators = NoiseRemovalDecorator(TranslationDecorator(base_classifier))  # Chain decorators
    df = noise_remover(df)
    # translate data to english
    df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df, classifier_with_decorators

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, classifier, name):
    classifier.train(data.X_train, data.y_train)
    for index, row in df.iterrows():
        email = Email(content=row[Config.INTERACTION_CONTENT], summary=row[Config.TICKET_SUMMARY])
        prediction = classifier.classify(email)  # Classify the email using the classifier with applied decorators
        print(f"Email {row[Config.TICKET_SUMMARY]} classified as: {prediction}")
    model_predict(data, df, name)
# Code will start executing from following line
if __name__ == '__main__':
    
    # pre-processing steps
    df = load_data()
    df, classifier = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    
    # data transformation
    X, group_df = get_embeddings(df)
    # data modelling
    data = get_data_object(X, df)
    # modelling
    perform_modelling(data, df, classifier, 'name')

