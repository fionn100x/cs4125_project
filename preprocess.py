#Methods related to data loading and all pre-processing steps will go here
import pandas as pd
import numpy as np
import re
from Config import Config

def get_input_data():
    purchasing_df = pd.read_csv('data/Purchasing.csv')
    app_gallery_df = pd.read_csv('data/AppGallery.csv')

    # combining the two datasets
    df = pd.concat([purchasing_df, app_gallery_df], ignore_index=True)
    return df

# removing duplicate entries
def de_duplication(df):
    df = df.drop_duplicates(subset=[Config.TICKET_SUMMARY, Config.INTERACTION_CONTENT])
    return df

def noise_remover(text: str) -> str:
    # Example of removing non-alphanumeric characters
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return clean_text

def translate_to_en(text_list):
    # Simulate translation by assuming text is already in English
    return text_list

