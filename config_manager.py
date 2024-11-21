import joblib
import threading

class ConfigurationManager:
    _instance = None  # Private class attribute to hold the single instance
    _lock = threading.Lock()  # Lock to ensure thread-safe initialization
    

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClassifierManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Load the pre-trained RandomForest model if it exists
        try:
            self.model = joblib.load('models/random_forest_classifier.pkl')
            print("RandomForest model loaded successfully in ClassifierManager.")
        except FileNotFoundError:
            self.model = None
            print("No pre-trained model found. Please train a model first.")

    def classify_email(self, data):
        if self.model:
            self.model.predict(data.get_X_test())
            data.test_df['Predicted Category'] = self.model.predictions
            return data.test_df[['Ticket id', 'Predicted Category']]
        else:
            raise RuntimeError("No model available. Please train the model before classification.")



    def __new__(cls, *args, **kwargs):
        """
        Overrides the __new__ method to ensure only one instance of ConfigurationManager is created.
        Uses a lock to make it thread-safe.
        """
        with cls._lock:
            if not cls._instance:
                cls._instance = super(ConfigurationManager, cls).__new__(cls)
                cls._instance._initialize_settings()
        return cls._instance

    def _initialize_settings(self):
        """
        Private method to set initial configuration values. 
        This method runs only once when the singleton instance is created.
        """
        self.database_path = "path/to/database.db"  # Path to the database
        self.api_key = "your-api-key"  # API key for external services
        self.email_categories = ["spam", "inbox", "promotion", "social"]  # Example categories
        self.logging_enabled = True  # Configuration for logging

    @staticmethod
    def get_instance():
        """
        Static method to retrieve the single instance of ConfigurationManager.
        Creates the instance if it doesn't already exist.
        """
        if not ConfigurationManager._instance:
            ConfigurationManager()  # Create the instance if it doesn't exist
        return ConfigurationManager._instance

    # Methods to access configuration settings
    def get_database_path(self):
        """
        Returns the path to the database.
        """
        return self.database_path

    def get_api_key(self):
        """
        Returns the API key for external services.
        """
        return self.api_key

    def get_email_categories(self):
        """
        Returns the list of email categories.
        """
        return self.email_categories

    def is_logging_enabled(self):
        """
        Returns whether logging is enabled.
        """
        return self.logging_enabled

    # Example method to update configuration
    def update_api_key(self, new_key):
        """
        Updates the API key with a new value.
        """
        self.api_key = new_key

    def toggle_logging(self):
        """
        Toggles the logging setting on or off.
        """
        self.logging_enabled = not self.logging_enabled