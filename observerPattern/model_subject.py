from observerPattern.observer import Observer
from .subject import Subject
from typing import List

class ModelSubject(Subject):
    """
    The Subject class maintains a list of observers and notifies them when events occur.
    """

    def __init__(self):
        self._observers: List[Observer] = []

    def attach(self, observer: Observer) -> None:
        print("ModelSubject: Attached an observer.")
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        print("ModelSubject: Detached an observer.")
        self._observers.remove(observer)

    def notify(self, event: str) -> None:
        """
        Notify all observers about an event.
        """
        for observer in self._observers:
            observer.update(event)
