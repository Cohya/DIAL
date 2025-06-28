import os 
import sys 
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import random
from simple_examples.nets import MessageNet, QNet
from simple_examples.guess_my_number import GuessMyNumberEnv
import numpy as np 
from learn2com.utils.discretise_regularise_unit import discretise_regularise_unit
import matplotlib.pyplot as plt

def dial_episode(i):
    if i % 100 == 0:
        q_target1.load_state_dict(q_net1.state_dict())
        q_target2.load_state_dict(q_net2.state_dict())

    obs1, obs2 = env.reset()
    total_reward1, total_reward2 = 0.0, 0.0

    history = []  # to store everything per step for backward pass
    msg2, msg1 = None, None 
    h1, h2 = None , None # GRU cell  inside 
    for step in range(env.max_steps):
        msg1, h1 = msg_net1(obs1, msg2, h1) # <- must be recurretn 
        msg2, h2 = msg_net2(obs2, msg1, h2) #< - must be recurrent  is not 10 rols and 1 are the same 
        msg1.retain_grad()
        msg2.retain_grad()

        msg1_after_dru = discretise_regularise_unit(msg1, scale=0.1, training=True)
        msg2_after_dru = discretise_regularise_unit(msg2, scale=0.1, training=True)
        q1 = q_net1(obs1, msg2_after_dru)
        q2 = q_net2(obs2, msg1_after_dru)

        a1 = q1.argmax(dim=-1)
        a2 = q2.argmax(dim=-1)

        obs_next, rewards, done = env.step(a1, a2)
        obs1_next, obs2_next = obs_next
        r1, r2 = rewards
        total_reward1 += torch.mean(r1).item()
        total_reward2 += torch.mean(r2).item()

        with torch.no_grad():
            next_msg1 = msg_net1(obs1_next)
            next_msg2 = msg_net2(obs2_next)
            next_msg1_dru = discretise_regularise_unit(next_msg1, scale=0.1, training=False)
            next_msg2_dru = discretise_regularise_unit(next_msg2, scale=0.1, training=False)
            q1_next = q_target1(obs1_next, next_msg2_dru)
            q2_next = q_target2(obs2_next, next_msg1_dru)
            max_q1 = q1_next.max(dim=1).values
            max_q2 = q2_next.max(dim=1).values
            target1 = r1 + gamma * max_q1
            target2 = r2 + gamma * max_q2

        q_val1 = q1.gather(1, a1.unsqueeze(1)).squeeze(1)
        q_val2 = q2.gather(1, a2.unsqueeze(1)).squeeze(1)
        td_error1 = (target1 - q_val1).detach()
        td_error2 = (target2 - q_val2).detach()

        q_loss = F.mse_loss(q_val1, target1) + F.mse_loss(q_val2, target2)

        history.append({
            'q_loss': q_loss,
            'td_error1': td_error1,
            'td_error2': td_error2,
            'msg1': msg1,
            'msg2': msg2
        })

        obs1, obs2 = obs1_next, obs2_next
        if done:
            break

    optimizer.zero_grad()

    # Step 1: backward Q loss for all steps
    total_loss = sum([step['q_loss'] for step in history])
    total_loss.backward(retain_graph=True)

    # Step 2: backward through messages (recursive from T to 0)
    mu1, mu2 = None, None
    for step in reversed(history):
        msg1 = step['msg1']
        msg2 = step['msg2']

        grad1 = step['td_error2'].unsqueeze(1) * msg1.grad.clone()
        grad2 = step['td_error1'].unsqueeze(1) * msg2.grad.clone()

        if mu1 is not None:
            grad1 = grad1 + mu1
            grad2 = grad2 + mu2

        msg_net1.zero_grad()
        msg_net2.zero_grad()
        msg1.backward(grad1, retain_graph=True)
        msg2.backward(grad2, retain_graph=True)

        mu1, mu2 = grad1.detach(), grad2.detach()

    optimizer.step()

    return total_loss.item(), total_reward1, total_reward2


# Run training
all_exp = dict()
env = GuessMyNumberEnv(batch_size=1, max_steps=4)
msg_dim = 1
msg_net1 = msg_net2 = MessageNet(msg_dim=msg_dim)
q_net1 = q_net2 = QNet(msg_dim=msg_dim)
q_target1 = q_target2 = QNet(msg_dim=msg_dim)

optimizer = Adam(
    list(msg_net1.parameters()) +
    list(q_net1.parameters()), lr=1e-3
)

gamma = 1.0
loss_vec, reward_vec = [], []
aveg_r, aveg_loss = [], []

for i in range(5000):
    loss, reward1, reward2 = dial_episode(i)
    reward_vec.append((reward1 + reward2) / 2)
    loss_vec.append(loss)
    if i % 100 == 0:
        print(f"Running episode {i}...")
        print(f"Average Reward: {np.mean(reward_vec[-100:]):.4f}, Loss: {np.mean(loss_vec[-100:]):.4f}")

    aveg_r.append(np.mean(reward_vec[-100:]))
    aveg_loss.append(np.mean(loss_vec[-100:]))
