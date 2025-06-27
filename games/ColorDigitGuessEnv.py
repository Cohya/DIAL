import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


class ColorDigitGuessEnv:
    def __init__(
        self,
        use_mnist=True,
        batch_size=1,
        game_colors=3,
        game_dim=28,
        game_level="extra_hard_local",
        nsteps=2,
        coop=0.0,
    ):
        self.use_mnist = use_mnist
        self.bs = batch_size
        self.game_colors = game_colors
        self.game_dim = game_dim
        self.game_level = game_level
        self.nsteps = nsteps
        self.step_counter = 1
        self.game_coop = coop
        self.action_space = 10  # digits 0–9

        self.reward = torch.zeros(self.bs, 2)
        self.terminal = torch.zeros(self.bs)

        self.mnist = self._load_mnist()
        self.obs_dims = (3, 28, 28)
        self.action_space = 10  # 0-9
        # TODO(Yaniv):check the level of the game
        self.info = {"digits": [], "colors": []}
        self.game_level = "easy"  # "extra_hard_local" or "easy"
        self.reset()

    def _load_mnist(self):
        dataset = torchvision.datasets.MNIST(root="./", train=True, download=True)
        data_by_digit = {i: [] for i in range(10)}
        for img, label in dataset:
            img_arr = np.array(img, dtype=np.float32) / 255.0
            data_by_digit[label].append(img_arr)
        return data_by_digit

    def _load_digit(self):
        x = torch.zeros(self.bs, self.game_colors, self.game_dim, self.game_dim)
        color_ids = torch.zeros(self.bs, dtype=torch.long)
        digits = torch.zeros(self.bs, dtype=torch.long)

        for b in range(self.bs):
            digit = random.randint(0, 9)
            color = random.randint(0, self.game_colors - 1)
            img = random.choice(self.mnist[digit])

            x[b, color] = torch.tensor(
                img
            )  # <-- here you say which color and actually clolor the image

            digits[b] = digit
            color_ids[b] = color

        return x, color_ids, digits

    def reset(self):
        self.state1 = self._load_digit()  # Agent 1
        self.state2 = self._load_digit()  # Agent 2
        self.step_counter = 1

        output = self.get_obs()
        self.info["digits"] = [self.state1[2], self.state2[2]]
        self.info["colors"] = [self.state1[1], self.state2[1]]

        return output

    def get_obs(self):
        return self.state1[0], self.state2[0]  # RGB images for both agents

    def get_reward(self, actions: list[torch.Tensor]):
        # actions: Tensor of shape (batch_size, 2)
        color_1 = self.state1[1]
        color_2 = self.state2[1]
        digit_1 = self.state1[2]
        digit_2 = self.state2[2]

        reward = torch.zeros(self.bs, 2)

        if self.game_level == "extra_hard_local" and self.step_counter > 1:
            a1, a2 = actions[0], actions[1]

            # Reward for agent 1
            term1 = digit_2 + a1 + color_1
            term2 = digit_1 + a1 + color_2
            reward[:, 0] = 2 * (-1) ** term1 + (-1) ** term2

            # Reward for agent 2
            term1 = digit_1 + a2 + color_2
            term2 = digit_2 + a2 + color_1
            reward[:, 1] = 2 * (-1) ** term1 + (-1) ** term2
        elif self.game_level == "easy" and self.step_counter > 1:
            a1, a2 = actions[0], actions[1]
            # Reward for agent 1
            term1 = digit_2 + a1
            # term2 = digit_1 + a1
            reward[:, 0] = 2 * (-1) ** term1  # + (-1) ** term2

            # Reward for agent 2
            term1 = digit_1 + a2  # + color_2
            # term2 = digit_2 + a2 + color_1
            reward[:, 1] = 2 * (-1) ** term1  # + (-1) ** term2

        # Cooperative reward mixing
        reward_coop = torch.zeros_like(reward)
        reward_coop[:, 0] = (reward[:, 0] + self.game_coop * reward[:, 1]) / (
            1 + self.game_coop
        )
        reward_coop[:, 1] = (reward[:, 1] + self.game_coop * reward[:, 0]) / (
            1 + self.game_coop
        )

        return reward_coop

    def get_max_reward(self):
        if self.step_counter > 1:
            color_1 = self.state1[1]
            color_2 = self.state2[1]
            digit_1 = self.state1[2]
            digit_2 = self.state2[2]
            actions = [digit_2, digit_1]
            reward_coop = self.get_reward(actions)
        else:
            reward_coop = torch.zeros(self.bs, 2)
        return reward_coop

    def step(self, actions: torch.Tensor):
        ## torch.stack(actions, dim=1)

        reward = self.get_reward(actions)
        terminal = (self.step_counter == self.nsteps) * torch.ones(self.bs)
        self.info["max_reward"] = self.get_max_reward()
        self.step_counter += 1
        
        return self.get_obs(), reward, terminal, self.info


# env = ColorDigitGuessEnv()
# obs1, obs2 = env.reset()
# done = False
# while not done:

#     actions = torch.randint(0, 10, (env.bs, 2))  # random actions
#     _, rewards, done = env.step(actions)
#     print(rewards[:5])
