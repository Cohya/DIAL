import torch 

def dru(m, sigma=2.0, training=True):
    if training:
        noise = torch.randn_like(m) * sigma
        return torch.sigmoid(m + noise)
    else:
        return (m > 0).float()

