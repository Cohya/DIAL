# Differentiable Communication (DIAL)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Reciver(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(Reciver, self).__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)

    def forward(self, messeage: torch.Tensor):
        # what is the dims that I need
        m = self.batch_norm(messeage)  # Normalize the input message
        x = self.fc1(m)
        return x


class LookUp(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LookUp, self).__init__()
        self.f = nn.Embedding(
            input_dim, output_dim
        )  # Assuming two agents, each represented by a unique index

    def forward(self, agent_index: torch.long):
        # agent_index = torch.tensor([0, 1], dtype=torch.long)  # batch of two agents
        # torch.long == torch.int64  # This is True
        # agent_index is a tensor containing the index of the agent
        agent_index = torch.tensor(agent_index, dtype=torch.long)
        return self.f(agent_index)


class TaskSpecificNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(TaskSpecificNet, self).__init__()
        input_dim = math.prod(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, obs: torch.Tensor):
        if len(obs.size()) > 2:
            obs = obs.view(obs.size(0), -1)

        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return x


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Input : (obs(t), m_a'(t-1) , h(t-1), u(t-1), a|theta), a is the agent, which agent is it !
class C_Net(nn.Module):

    def __init__(
        self,
        obs_dims: int,
        number_of_agents: int,
        action_dims: int,
        message_dims: int,
        embedding_dim: int,
    ):
        self.action_dims = action_dims
        self.message_dims = message_dims
        self.obs_dims = obs_dims
        super(C_Net, self).__init__()
        self.reciver = Reciver(
            input_dim=message_dims, hidden_dim=embedding_dim
        )  # embbeding the message to 128
        self.lookUpAgent = LookUp(input_dim=number_of_agents, output_dim=embedding_dim)

        self.task_specific_net = TaskSpecificNet(
            input_dim=obs_dims, hidden_dim=embedding_dim, output_dim=embedding_dim
        )

        self.LookUpAction = LookUp(input_dim=action_dims, output_dim=embedding_dim)

        self.gru1 = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )
        self.gru2 = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )

        self.mlp_layer = MLPNet(
            input_dim=128, hidden_dim=128, output_dim=(action_dims + message_dims)
        )

    def forward(
        self,
        obs: torch.Tensor,
        message: torch.Tensor,
        u_tm1: int,
        a: torch.long,
        h_1: torch.Tensor,
        h_2: torch.Tensor,
    ):
        # obs: observation of the agent
        # message: message from the other agent
        # agent_index: index of the agent

        # Identify the agent
        # a dims list of int --> (B, 128)
        agend_id_embedded = self.lookUpAgent(a)

        # Process the message

        if (len(message) == 0 and message is not None) or not torch.isnan(
            message
        ).any():
            message_embedding = self.reciver(message)
        else:
            message_embedding = torch.zeros_like(agend_id_embedded)
        # message_embedding  --> (B, 128)
        # Process the observation
        o_t_enbedded = self.task_specific_net(obs)

        # Embed the action (u_tm1 == action at time t-1)
        if (len(u_tm1) == 0 and u_tm1 is not None) or not torch.isnan(u_tm1).any():
            action_embedding = self.LookUpAction(u_tm1)
        else:
            action_embedding = torch.zeros_like(agend_id_embedded)

        # Combine observation, message embedding, and agent embedding
        z_t_a = o_t_enbedded + message_embedding + agend_id_embedded + action_embedding

        # Reshape z_t_a for GRU input
        z_t_a = z_t_a.unsqueeze(1)  # (B, 1, emmbedding_dim)
        # if h_1 is None:  # torch.randn(1, batch_size, 128)
        #     h_1 = torch.zeros(1, z_t_a.size(0), 128).to(
        #         z_t_a.device
        #     )  # (num_layers, batch_size, hidden_size)
        # else:
        h_1 = h_1.to(z_t_a.device)

        # if h_2 is None:
        #     h_2 = torch.zeros(1, z_t_a.size(0), 128).to(
        #         z_t_a.device
        #     )  # (num_layers, batch_size, hidden_size)
        # else:
        h_2 = h_2.to(z_t_a.device)

        out1, h1 = self.gru1(z_t_a, h_1)  # Add batch dimension
        out2, h2 = self.gru2(out1, h_2)  # h1[0] is the output of the GRU
        out2 = out2.squeeze(1)  # bring
        UM = self.mlp_layer(out2)
        q_values = UM[:, : self.action_dims]
        message = UM[:, self.action_dims :]
        return q_values, message, out2, h1, h2

    def get_weights(self):
        return self.parameters()

    def copy_weights_from_other_network(self, other_network: torch.nn.Module):
        for param, other_param in zip(self.parameters(), other_network.parameters()):
            param.data.copy_(other_param.data)


if __name__ == "__main__":
    cnet = C_Net(obs_dims=10, number_of_agents=2, action_dims=5, embedding_dim=128)
