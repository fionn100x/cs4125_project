from model.randomforest import RandomForest
from model.classification_strategy import (
    ClassifierContext, 
    RandomForestStrategy,
    AdaBoostStrategy,
    HistGradientBoostingStrategy
)

def model_predict(data, df, name, subject):
    # Initialize strategies
    strategies = {
        'randomforest': RandomForestStrategy(),
        'adaboost': AdaBoostStrategy(),
        'histgradientboosting': HistGradientBoostingStrategy()
    }
    
    # Select strategy based on name
    strategy = strategies.get(name.lower(), RandomForestStrategy())
    classifier = ClassifierContext(strategy, subject)
    
    # Train
    classifier.train_classifier(data.X_train, data.y_train)
    
    # Predict
    predictions = classifier.classify(data.X_test)

    # Notify observers
    for pred in predictions:
        subject.notify_observers(pred)
    
    # Print results
    from sklearn.metrics import classification_report
    print(classification_report(data.y_test, predictions))


def model_evaluate(model, data):
    model.print_results(data)