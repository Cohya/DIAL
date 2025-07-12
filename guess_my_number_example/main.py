import os 
import sys 

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from guess_my_number_example.networks import AgentNet
from guess_my_number_example.guess_my_number import GuessMyNumberEnv
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle 
from guess_my_number_example.playfull_episode import play_full_episode

from guess_my_number_example.dial_algorithm import apply_dial_algorithm


input_dim = 1
hidden_dim = 32
msg_dim = 1
action_dim = 5
max_steps = 5 #num of bits are 5-1

agent1 = AgentNet(input_dim + msg_dim, hidden_dim, msg_dim, action_dim)
agent2 = AgentNet(input_dim + msg_dim, hidden_dim, msg_dim, action_dim)
agnet_1_target = AgentNet(input_dim + msg_dim, hidden_dim, msg_dim, action_dim)
# Share weights 
agent2.load_state_dict(agent1.state_dict())
agnet_1_target.load_state_dict(agent1.state_dict())
optim = torch.optim.Adam(list(agent1.parameters()) + list(agent2.parameters()), lr=1e-4)#was 3

env = GuessMyNumberEnv(max_steps = max_steps, action_space = action_dim)
gamma = 0.9
loss_vec = []
average_r = []
for episode in range(100000):


    agent_1_record, agent_2_record , avege_reward = play_full_episode(env, agent1, agent2, optim, gamma, hidden_dim)

    average_r.append(avege_reward)

    loss,gradients_agent = apply_dial_algorithm(agent_1_record, agent_2_record, agent1, agent2, optim, gamma, hidden_dim, agnet_1_target)

    loss_vec.append(loss)
    ## lets applay the gradient to the network 
    gradint_agent_1 = gradients_agent[0]
    gradint_agent_2 = gradients_agent[1]
    optim.zero_grad()
    for param,grad, grad2 in zip(agent1.parameters(), gradint_agent_1, gradint_agent_2):
        gradient_of_param = (grad + grad2) / 2
        if gradient_of_param is None:
            continue
        param.grad = gradient_of_param


    optim.step()

    # copy weights
    agent2.load_state_dict(agent1.state_dict())
    if episode % 100 == 0:
        agnet_1_target.load_state_dict(agent1.state_dict())

    if episode % 100 == 0:
        print("episode: ", episode, "average reward: ", np.mean(average_r[-100:]), "loss: ", np.mean(loss_vec[-100:]))

with open("loss_vec.pk", "wb") as file:
    pickle.dump(loss_vec, file)

with open("average_r.pk", "wb") as file:
    pickle.dump(average_r,file)


# smooth the averag_r and create a plot 

average_r = np.array(average_r)
average_r = np.convolve(average_r, np.ones(100)/100, mode='valid')
plt.plot(average_r)
#save the iage 
plt.savefig("average_r.png")
# cerate the same for the loss_vec 
loss_vec = np.array(loss_vec)
loss_vec = np.convolve(loss_vec, np.ones(100)/100, mode='valid')
plt.plot(loss_vec)
plt.savefig("loss_vec.png")