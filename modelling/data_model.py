from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class Data:
    def __init__(self, X: np.ndarray, vectorizer, df: pd.DataFrame) -> None:
        self.vectorizer = vectorizer
        self.df = df  # Store the entire DataFrame

        # Perform train-test split and keep indices
        train_indices, test_indices = train_test_split(
            df.index, test_size=0.2, random_state=0, stratify=df['y2']
        )

        # Split features and labels
        self.X_train = X[train_indices]
        self.X_test = X[test_indices]
        self.y_train = df.loc[train_indices, 'y2'].values
        self.y_test = df.loc[test_indices, 'y2'].values

        # Split raw text data
        self.df_train = df.loc[train_indices]
        self.df_test = df.loc[test_indices]

    def get_type(self):
        return self.y_train

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_type_y_train(self):
        return self.y_train

    def get_type_y_test(self):
        return self.y_test

    def get_train_df(self):
        return self.train_df

    def get_embeddings(self):
        return self.embeddings

    def get_type_test_df(self):
        return self.test_df
