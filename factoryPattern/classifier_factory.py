import model.classification_strategy


class ClassifierFactory:
    @staticmethod
    def create_classifier(strategy_type: str):
        if strategy_type == "randomforest":
            return model.classification_strategy.RandomForestStrategy()
        elif strategy_type == "adaboost":
            return model.classification_strategy.AdaBoostStrategy()
        elif strategy_type == "histgb":
            return model.classification_strategy.HistGradientBoostingStrategy()
        elif strategy_type == "sgd":
            return model.classification_strategy.SGDStrategy()
        elif strategy_type == "voting":
            return model.classification_strategy.VotingStrategy()
        else:
            raise ValueError(f"Unknown classifier strategy: {strategy_type}")

    @staticmethod
    def get_supported_models():
        """Return a list of all supported classifier names."""
        return ["randomforest", "adaboost", "histgb", "sgd", "voting"]