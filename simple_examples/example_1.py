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
# ─────────────────────────────────────────────
# Dummy 2-agent environment
# ─────────────────────────────────────────────

from simple_examples.guess_my_number import GuessMyNumberEnv

import numpy as np 
from learn2com.utils.discretise_regularise_unit import discretise_regularise_unit

env = GuessMyNumberEnv(batch_size = 1, max_steps = 4)
msg_dim = 1
msg_net1 = MessageNet(msg_dim = msg_dim)
msg_net2 = MessageNet(msg_dim = msg_dim)
q_net1 = QNet(msg_dim = msg_dim)
q_net2 = QNet(msg_dim = msg_dim)

q_target1 = QNet(msg_dim = msg_dim)
q_target2 = QNet(msg_dim = msg_dim)

optimizer = Adam(
        list(msg_net1.parameters()) +
        list(msg_net2.parameters()) +
        list(q_net1.parameters()) +
        list(q_net2.parameters()), lr=1e-3
    )
gamma = 0.99

def dial_episode(i):
    if i % 100 == 0:
        q_target1.load_state_dict(q_net1.state_dict())
        q_target2.load_state_dict(q_net2.state_dict())


  
    obs1, obs2 = env.reset()
    total_reward1, total_reward2 = 0.0, 0.0
    mu1, mu2 = None, None
    for step in range(env.max_steps):
        msg1 = msg_net1(obs1)
        msg2 = msg_net2(obs2)
        msg1.retain_grad()
        msg2.retain_grad()

        msg1_after_dru = discretise_regularise_unit(msg1, scale=0.1, training=True)
        msg2_after_dru = discretise_regularise_unit(msg2, scale=0.1, training=True)
        q1 = q_net1(obs1, msg2_after_dru)
        q2 = q_net2(obs2, msg1_after_dru)

        a1 = q1.argmax(dim=-1)
        a2 = q2.argmax(dim=-1)

        # r1, r2 = env.get_reward(a1, a2)

        obs_next,rewards, done = env.step(a1, a2)
        obs1_next, obs2_next = obs_next
        r1, r2  = rewards
        total_reward1 += torch.mean(r1).item()
        total_reward2 += torch.mean(r2).item()
        
        with torch.no_grad():
            next_msg1 = msg_net1(obs1_next).detach()
            next_msg2 = msg_net2(obs2_next).detach()

            next_msg1_after_dru = discretise_regularise_unit(next_msg1, scale=0.1, training=False)
            next_msg2_after_dru = discretise_regularise_unit(next_msg2, scale=0.1, training=False)

            next_q1 = q_target1(obs1_next, next_msg2_after_dru)
            next_q2 = q_target2(obs2_next, next_msg1_after_dru)

            max_q1 = next_q1.max(dim=1).values
            max_q2 = next_q2.max(dim=1).values

            target1 = r1 + gamma * max_q1
            target2 = r2 + gamma * max_q2

        q_val1 = q1.gather(1, a1.unsqueeze(1)).squeeze(1)
        q_val2 = q2.gather(1, a2.unsqueeze(1)).squeeze(1)

        td_error1 = (target1 - q_val1).detach()
        td_error2 = (target2 - q_val2).detach()

        # Step 1: backprop Q loss
        q_loss = F.mse_loss(q_val1, target1) + F.mse_loss(q_val2, target2)
        optimizer.zero_grad()
        q_loss.backward(retain_graph=True)

        # Step 2: recursive μ update

        # Step 2: backprop through messages using μ(t+1)
        mu1_new = td_error2.unsqueeze(1) * msg1.grad.clone()
        mu2_new = td_error1.unsqueeze(1) * msg2.grad.clone()

        if mu1 is not None:
            msg_net1.zero_grad()
            msg_net2.zero_grad()
            msg1.backward(mu1, retain_graph=True)
            msg2.backward(mu2, retain_graph=True)
        
        mu1, mu2 = mu1_new, mu2_new
        optimizer.step()

        obs1, obs2 = obs1_next, obs2_next
        if done:
            break

    return q_loss.item(), total_reward1, total_reward2


# Run 1 episode
loss_vec  = []
reward_vec = []
for i in range(10000):
    loss, reward1, reward2 = dial_episode(i)
    reward_vec.append((reward1+ reward2 )/ 2)
    loss_vec.append(loss)
    if i % 100 == 0:
        print(f"Running episode {i}...")
        # print(f"episode: {i}, Loss: {loss:.4f}, Total Reward1: {reward1}, Total Reward2: {reward2}")
        print(f"Average Reward: {np.mean(reward_vec[-100:]):.4f}, Loss: {np.mean(loss_vec[-100:]):.4f}")