import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Update gate
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)

        # Reset gate
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)

        # Candidate hidden state
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h_prev):
        z = torch.sigmoid(self.W_z(x) + self.U_z(h_prev))  # update gate
        r = torch.sigmoid(self.W_r(x) + self.U_r(h_prev))  # reset gate
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(r * h_prev))  # candidate
        h = (1 - z) * h_prev + z * h_tilde  # final hidden state
        return h
