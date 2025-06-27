import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from abc import ABC, abstractmethod
from typing import Any

from nn_models.learn2com.agents.abstract_agent import AgentAbstract
from nn_models.learn2com.methods.ddqn import RL_model
from nn_models.learn2com.utils.eps_greedy import epsilone_greedy
from nn_models.learn2com.utils.rb_abstract import RBAbstract


class Agent(AgentAbstract):
    def __init__(self, model: RL_model, i_d, replay_buffer: RBAbstract):
        self.model = model
        self.i_d = i_d
        self.replay_buffer = replay_buffer
        self.action_dims = model.main_net.action_dims

    def add_to_replay_buffer(self, **kwargs):
        """Add experience to the replay buffer."""
        return self.replay_buffer.push(**kwargs)

    def get_action_and_message(self, eps: float, **kwargs: Any):
        """Return an action based on the current state."""
        output = self.model.get_action_and_message(**kwargs)
        q_values, message, out2, h1, h2 = output
        # q_values = output[:, : self.model.action_dims]
        # message = output[:, self.action_dims :]
        action = epsilone_greedy(epsilon=eps, q_values=q_values)
        return action, message, out2, h1, h2

    def learn(self, *args):
        """Update the agent's knowledge based on the experience."""
        return self.model.learn(*args)

    def save(self, path):
        """Save the agent's model or state."""
        return self.model.save(path)

    def load(self, path):
        """Load the agent's model or state."""
        return self.model.load(path)
