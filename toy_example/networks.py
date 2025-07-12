import torch.nn as nn

class AgentNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, msg_dim, action_dim):
        super().__init__()
        self.rnn = nn.GRUCell(input_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, action_dim)
        self.msg_head = nn.Linear(hidden_dim, msg_dim)

    def forward(self, x, h):
        h = self.rnn(x, h)
        q = self.q_head(h)
        m = self.msg_head(h)
        return q, m, h