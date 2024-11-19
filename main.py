from modelling.modelling import model_predict
from preprocess import *
from embeddings import *
from modelling import *
from modelling.data_model import *
import random
import numpy as np


seed = 0
random.seed(seed)
np.random.seed(seed)

def load_data():
    df = pd.read_csv("data/AppGallery.csv")  # Load dataset
    return df

def preprocess_data(df):
    df = de_duplication(df)
    df = noise_remover(df)
    df['Ticket Summary'] = translate_to_en(df['Ticket Summary'].tolist())
    df['y2'] = df['Type 2']
    return df

def perform_modelling(data, df, name):
    model_predict(data, df, name)

def get_embeddings(df):
    X, vectorizer = get_tfidf_embd(df)
    return X, vectorizer, df

def get_data_object(X, vectorizer, df):
    return Data(X, vectorizer, df)

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    df['Interaction content'] = df['Interaction content'].astype(str)
    df['Ticket Summary'] = df['Ticket Summary'].astype(str)

    model_name = input("Select model (randomforest/adaboost/histgb): ").lower()

    X, vectorizer, group_df = get_embeddings(df)
    data = get_data_object(X, vectorizer, df)
    perform_modelling(data, df, model_name)
