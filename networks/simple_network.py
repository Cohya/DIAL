import torch.nn as nn
import  torch 
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
    

class  AgentNet2(nn.Module):
    def __init__(self, input_dim, hidden_dim, msg_dim, action_dim):
        super().__init__()
        self.embedding_layer = nn.Embedding(action_dim, hidden_dim)
        self.embedding_message = nn.Linear(msg_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim*2, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, action_dim)
        self.msg_head = nn.Linear(hidden_dim, msg_dim)

    def forward(self, x, h):
        x_obs = self.embedding_layer(x[:,:-1].to(torch.long)).squeeze(1)  # Exclude the last element for message
        x_msg = self.embedding_message(x[:, -1:])  # Last element is the message
        x =  torch.cat((x_obs, x_msg), dim=-1)
        h = self.rnn(x, h)
        q = self.q_head(h)
        m = self.msg_head(h)
        return q, m, h