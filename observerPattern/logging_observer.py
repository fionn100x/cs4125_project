from .observer import Observer

class LoggingObserver(Observer):
    def update(self, event: str) -> None:
        print(f"LoggingObserver: Event logged: {event}")
