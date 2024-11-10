#Methods related to converting text in into numeric representation and then returning numeric representation may go here
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from Config import Config

def get_tfidf_embd(df):
    """Get TF-IDF embeddings from text"""
    vectorizer = TfidfVectorizer(
        max_features=1000,  # Limit features to top 1000
        stop_words='english',
        ngram_range=(1, 2)  # Use both unigrams and bigrams
    )
    
    # combining ticket summary and content for embedding
    text_data = df[Config.TICKET_SUMMARY] + " " + df[Config.INTERACTION_CONTENT]
    
    # generating the  TF-IDF matrix
    X = vectorizer.fit_transform(text_data)
    
    return X.toarray()