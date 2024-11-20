from model.randomforest import RandomForest
from model.classification_strategy import (
    ClassifierContext, 
    RandomForestStrategy,
    AdaBoostStrategy,
    VotingStrategy  
)

def model_predict(data, df, name):
    strategies = {
        'randomforest': RandomForestStrategy(),
        'adaboost': AdaBoostStrategy(),
        'voting': VotingStrategy()  
    }
    
    
    strategy = strategies.get(name.lower(), RandomForestStrategy())  # Default to RandomForest if no match
    classifier = ClassifierContext(strategy)
    
    
    classifier.train_classifier(data.X_train, data.y_train)
    
    
    predictions = classifier.classify(data.X_test)
    
    
    from sklearn.metrics import classification_report
    print(classification_report(data.y_test, predictions))
