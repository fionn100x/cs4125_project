import threading
import joblib


class ModelManager:
    """
    Singleton class for managing multiple machine learning models.
    """
    _instance = None
    _lock = threading.Lock()  # Lock for thread safety

    def __new__(cls, *args, **kwargs):
        """
        Singleton instance creation with thread safety.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Initialize the model manager and load pre-trained models if they exist.
        """
        self.models = {}  # Dictionary to store model instances by name
        self.model_files = {
            "randomforest": "models/random_forest_classifier.pkl",
            "adaboost": "models/adaboost_classifier.pkl",
            "histgb": "models/histgb_classifier.pkl",
            "voting": "models/voting_classifier.pkl",
        }
        self._load_models()

    def _load_models(self):
        """
        Load pre-trained models from disk into memory.
        """
        for model_name, model_file in self.model_files.items():
            try:
                self.models[model_name] = joblib.load(model_file)
                print(f"Model '{model_name}' loaded successfully.")
            except FileNotFoundError:
                self.models[model_name] = None
                print(f"No pre-trained model found for '{model_name}'. Please train it first.")

    def get_model(self, model_name):
        """
        Get the model instance for the given model name.
        """
        model = self.models.get(model_name)
        if model is None:
            raise RuntimeError(f"Model '{model_name}' is not available. Please train it first.")
        return model

    def save_model(self, model_name, model):
        """
        Save a trained model to disk.
        """
        if model_name not in self.model_files:
            raise ValueError(f"Unknown model name '{model_name}'.")
        joblib.dump(model, self.model_files[model_name])
        self.models[model_name] = model
        print(f"Model '{model_name}' saved successfully.")

    def classify(self, model_name, data):
        """
        Use the specified model to classify the given data.
        """
        model = self.get_model(model_name)
        predictions = model.predict(data.get_X_test())
        data.test_df["Predicted Category"] = predictions
        return data.test_df[["Ticket id", "Predicted Category"]]


# Example Usage:
# Access the Singleton instance
#model_manager = ModelManager()

