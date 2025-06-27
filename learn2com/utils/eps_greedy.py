import numpy as np
import torch


def epsilone_greedy(epsilon: float, q_values: torch.Tensor):
    """
    Selects an action using epsilon-greedy strategy.

    Args:
        epsilon (float): Probability of selecting a random action.
        action_space (list): List of possible actions.
        q_values (list): Q-values for each action.

    Returns:
        int: Selected action index.
    """
    if np.random.rand() < epsilon:
        _, actions_dims = q_values.size()
        return torch.Tensor(
            np.random.choice(actions_dims, size=(q_values.shape[0],))
        )  # Explore
    else:
        return torch.argmax(q_values, axis=1)  # Exploit
