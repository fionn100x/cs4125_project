import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from model.email import Email
from sklearn.metrics import classification_report
from observerPattern.model_subject import ModelSubject

class VotingModel(ModelSubject):
    def __init__(self, model_name: str, embeddings: np.ndarray, y: np.ndarray, vectorizer) -> None:
        super().__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.vectorizer = vectorizer
        self.model = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier()),
                ('ada', AdaBoostClassifier()),
                ('hist', HistGradientBoostingClassifier())
            ],
            voting='soft'  # Use 'soft' voting for probability-based combination
        )
        self.predictions = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.notify("Voting Training Started")
        self.model.fit(X, y)
        self.notify("Voting Training Completed")

    def predict(self, content: str):
        self.notify("Voting Predicting Email")
        email = Email(content=content, summary="")
        features = email.to_features(self.vectorizer)
        prediction = self.model.predict([features])
        self.notify("Voting Email Predicted")
        return prediction

    def print_results(self, data):
        self.notify("Voting Printing Results")
        predictions = self.model.predict(data.get_X_test())
        print(classification_report(data.get_type_y_test(), predictions))
        self.notify("Voting Results Printed")

    def data_transform(self) -> None:
        self.embeddings, self.labels = self.embeddings, self.y
