from abc import ABC, abstractmethod


class RBAbstract(ABC):
    @abstractmethod
    def push(self, *args):
        """Push a new experience into the replay buffer."""
        pass

    @abstractmethod
    def sample(self, *args):
        """Sample a batch of experiences from the replay buffer."""
        pass

    @abstractmethod
    def __len__(self):
        """Return the current size of the replay buffer."""
        pass
