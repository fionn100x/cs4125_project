import pandas as pd
from Config import Config



def get_input_data():
    """Loads and combines input data from CSV files."""
    purchasing_df = pd.read_csv('data/Purchasing.csv')
    app_gallery_df = pd.read_csv('data/AppGallery.csv')
    df = pd.concat([purchasing_df, app_gallery_df], ignore_index=True)
    return df


def de_duplication(df):
    """Removes duplicate rows from the DataFrame."""
    return df.drop_duplicates()

def noise_remover(df):
    """Removes common noise words or patterns and unwanted characters from text columns."""
    # Removing general noise like punctuation and converting text to lowercase
    df['Interaction content'] = df['Interaction content'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    df['Ticket Summary'] = df['Ticket Summary'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

    # Removing specific noise patterns
    noise_patterns = r'\b(thank you|please|regards|sincerely|best regards)\b'
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].str.replace(noise_patterns, '', regex=True)
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].str.replace(noise_patterns, '', regex=True)
    
    return df

def translate_to_en(text_list):
    """Simulates the translation of text to English."""
    # This mock function simply returns the input text for now; replace with an API/library for actual translation
    return text_list
