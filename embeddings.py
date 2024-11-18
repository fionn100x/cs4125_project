from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_embd(df):
    vectorizer = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.9)
    X = vectorizer.fit_transform(df['Interaction content']).toarray()
    return X
