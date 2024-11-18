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

def get_embeddings(df):
    X = get_tfidf_embd(df)
    return X, df

def get_data_object(X, df):
    return Data(X, df)

def perform_modelling(data, df, name):
    model_predict(data, df, name)

if __name__ == '__main__':

    df = load_data()
    df = preprocess_data(df)
    df['Interaction content'] = df['Interaction content'].values.astype('U')
    df['Ticket Summary'] = df['Ticket Summary'].values.astype('U')

    X, group_df = get_embeddings(df)
    data = get_data_object(X, df)
    perform_modelling(data, df, 'RandomForestModel')
