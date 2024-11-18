#Methods related to data loading and all pre-processing steps will go here
import pandas as pd
import numpy as np
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

def noise_remover(df):
    """Removes common noise words or patterns from text columns."""
    noise_patterns = r'\b(thank you|please|regards|sincerely|best regards)\b'
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].str.replace(noise_patterns, '', regex=True)
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].str.replace(noise_patterns, '', regex=True)
    return df

# Example placeholder for translation, mock function for translation to English
def translate_to_en(text_list):
    """Simulates the translation of text to English."""
    # This mock function simply returns the input text for now
    return text_list