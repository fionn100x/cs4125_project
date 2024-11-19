from .observer import Observer

class LoggingObserver(Observer):

    def __init__(self, observer_name: str):
        self.observer_name = observer_name  # Unique name for the logging observer

    def update(self, event: str) -> None:
        print(f"{self.observer_name}: Event logged: {event}")
