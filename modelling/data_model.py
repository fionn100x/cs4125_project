from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class Data:
    def __init__(self, X: np.ndarray, df: pd.DataFrame) -> None:
        # Perform train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, df['y2'].values, test_size=0.2, random_state=0, stratify=df['y2']
        )

        # Get train-test indices using DataFrame index
        train_mask = np.isin(X, self.X_train, assume_unique=False)
        test_mask = ~train_mask

        self.train_df = df.iloc[train_mask]
        self.test_df = df.iloc[test_mask]
        self.embeddings = X

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
