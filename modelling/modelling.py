# modelling.py
from model.classification_strategy import (
    AdaBoostStrategy,
    RandomForestStrategy,
    HistGradientBoostingStrategy,
    SGDStrategy,
    ClassifierContext
)
from model.email import Email
from sklearn.metrics import classification_report

def model_predict(data, df, name: str):
    strategies = {
        'adaboost': AdaBoostStrategy(),
        'randomforest': RandomForestStrategy(),
        'histgb': HistGradientBoostingStrategy(),
        'sgd': SGDStrategy()
    }

    strategy = strategies.get(name.lower())
    if not strategy:
        raise ValueError(f"Unknown model: {name}. Available models: {list(strategies.keys())}")

    classifier = ClassifierContext(strategy)

    # Initialize and train the model
    classifier.train_classifier(data.get_X_train(), data.get_type_y_train(), data.vectorizer)

    # Create email objects from test data
    predictions = []
    for _, row in data.df_test.iterrows():
        content = row['Interaction content']
        summary = row['Ticket Summary']
        email = Email(content=str(content), summary=str(summary))
        pred = classifier.classify_email(email)
        predictions.append(pred[0])  # Get the prediction from the array

    # Print results
    strategy.model.print_results(data)

    return predictions