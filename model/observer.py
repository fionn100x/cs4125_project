from abc import ABC, abstractmethod

class Observer(ABC):
    """
    Abstract base class for all observers in the Observer Design Pattern.
    Each observer must implement the `update` method.
    """
    @abstractmethod
    def update(self, data):
        """
        Receive updates from the subject.
        :param data: The data being passed to the observer (e.g., classification result).
        """
        pass

class Subject:
    """
    Subject base class that manages observers and notifies them of updates.
    """
    def __init__(self):
        self._observers = []  # List of observers

    def register_observer(self, observer: Observer):
        """
        Register an observer to receive updates.
        :param observer: Instance of a class that implements the Observer interface.
        """
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        """
        Remove an observer from the notification list.
        :param observer: Instance of a class that implements the Observer interface.
        """
        if observer in self._observers:
            self._observers.remove(observer)

    def notify_observers(self, data):
        """
        Notify all registered observers with the provided data.
        :param data: The data to send to observers.
        """
        for observer in self._observers:
            observer.update(data)

class ResultDisplayer(Observer):
    """
    A concrete observer that displays classification results.
    """
    def update(self, data):
        print(f"[ResultDisplayer] New classification result: {data}")

class StatisticsLogger(Observer):
    """
    A concrete observer that logs classification statistics.
    """
    def update(self, data):
        print(f"[StatisticsLogger] Logging classification result: {data}")
