# This action network is used to predict the next action based on: Q_a(obs(t), m(t-1) , h(t-1), u(t)) where u in U and m in M
# m are messages from other agents !
import torch
import torch.nn as nn


class ActionValueNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActionValueNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
