from model.randomforest import RandomForest
from model.hist_gb import HistGradientBoosting
from observerPattern.logging_observer import LoggingObserver
def model_predict(data, df, name):
    #comment out based on what model you want to run (this is where we used classification design pattern
    model = RandomForest(name, data.get_embeddings(), data.get_type())
    #model = HistGradientBoosting(name, data.get_embeddings(), data.get_type())
    #adding observer to model subject
    logger = LoggingObserver()
    model.attach(logger)
    model.train(data)
    model.predict(data.get_X_test())
    model_evaluate(model, data)

def model_evaluate(model, data):
    model.print_results(data)
