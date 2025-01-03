# data_model.py
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class Data:
    def __init__(self, X: np.ndarray, vectorizer, df: pd.DataFrame, target_column: str = 'y2') -> None:
        self.vectorizer = vectorizer
        self.df = df
        self.target_column = target_column
        self.embeddings = X  # Store embeddings

        # Validate inputs
        if len(X) != len(df):
            raise ValueError("The number of rows in X and df must match.")
        if self.target_column not in df.columns:
            raise KeyError(f"The target column '{self.target_column}' is not in the DataFrame.")
        
        # Perform train-test split and keep indices
        train_indices, test_indices = train_test_split(
            df.index, test_size=0.2, random_state=0, stratify=df[self.target_column].dropna()
        )

        # Split features and labels
        self.X_train = X[train_indices]
        self.X_test = X[test_indices]
        self.y_train = df.loc[train_indices, self.target_column].values
        self.y_test = df.loc[test_indices, self.target_column].values

        # Split raw text data
        self.df_train = df.loc[train_indices]
        self.df_test = df.loc[test_indices]

    def get_type(self):
        """Returns all unique types in the target column."""
        return self.df[self.target_column].unique()

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_type_y_train(self):
        return self.y_train

    def get_type_y_test(self):
        return self.y_test

    def get_train_df(self):
        return self.df_train

    def get_test_df(self):
        return self.df_test

    def get_embeddings(self):
        return self.embeddings
