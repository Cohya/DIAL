import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from abc import ABC, abstractmethod


class AgentAbstract(ABC):
    @abstractmethod
    def get_action_and_message(self, *args):
        """Return an action based on the current state."""
        pass

    @abstractmethod
    def learn(self, *args):
        """Update the agent's knowledge based on the experience."""
        pass

    @abstractmethod
    def save(self, *args):
        """Save the agent's model or state."""
        pass

    @abstractmethod
    def load(self, *args):
        """Load the agent's model or state."""
        pass
