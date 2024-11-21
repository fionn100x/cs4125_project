from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_embd(df):
    """Get TF-IDF embeddings from text."""
    vectorizer = TfidfVectorizer(
        max_features=1000,  # Limit features to top 1000
        stop_words='english',
        ngram_range=(1, 2)  # Use both unigrams and bigrams
    )
    
    # Combine ticket summary and content for embedding
    text_data = df['Ticket Summary'] + " " + df['Interaction content']
    
    # Generate the TF-IDF matrix
    X_sparse = vectorizer.fit_transform(text_data)
    
    # Convert sparse matrix to dense
    X = X_sparse.toarray()
    
    return X, vectorizer
