from abc import ABC, abstractmethod

# Observer Interface
class Observer(ABC):
    @abstractmethod
    def update(self, event: str) -> None:
        """
        Receive an update from the subject.
        """
        pass
