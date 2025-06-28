import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# Simple agent networks
# ─────────────────────────────────────────────
class MessageNet(nn.Module):
    def __init__(self, msg_dim=1):
        super().__init__()
        self.obs_proj =  nn.Embedding(
            10, 128
        ) 
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, msg_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.long)
        x = self.obs_proj(x)
        return self.net(x)



class MessageNetRecuurent(nn.Module):
    def __init__(self, msg_dim=1):
        super().__init__()
        self.obs_proj =  nn.Embedding(
            10, 128
        ) 

        self.gru1 = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, msg_dim),
            nn.Tanh()
        )

    def forward(self, x,  h):
        x = torch.tensor(x, dtype=torch.long)
        x = self.obs_proj(x)
        x,h = self.gru1(x)  # Add batch dimension
        return self.net(x), h 
    

class QNet(nn.Module):
    def __init__(self, msg_dim):
        super().__init__()
        obs_emb =64  # Embedding size for observations
        self.obs_proj =  nn.Embedding(
            10, obs_emb
        ) 

        self.message_proj = nn.Linear(msg_dim, obs_emb)
        self.fc = nn.Sequential(
            nn.Linear(obs_emb + obs_emb, 128),
            nn.ReLU(),
            nn.Linear(128, 10),  # Assuming action_space is defined elsewhere
            nn.Sigmoid()
        )

    def forward(self, obs, msg):
        obs = torch.tensor(obs, dtype=torch.long)
        obs_feat = self.obs_proj(obs)
        emb_msg = self.message_proj(msg)
        x = torch.cat([obs_feat, emb_msg], dim=-1)
        return self.fc(x)
