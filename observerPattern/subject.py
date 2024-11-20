from abc import ABC, abstractmethod
from .observer import Observer


# Subject Interface
class Subject(ABC):
    @abstractmethod
    def attach(self, observer: Observer) -> None:
        """
        Attach an observer to the subject.
        """
        pass

    @abstractmethod
    def detach(self, observer: Observer) -> None:
        """
        Detach an observer from the subject.
        """
        pass

    @abstractmethod
    def notify(self, event: str) -> None:
        """
        Notify all observers about an event.
        """
        pass
