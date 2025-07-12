import os 
import sys 

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from toy_example.networks import AgentNet
from toy_example.guess_my_number import GuessMyNumberEnv
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle 

# ==== DRU (Discretize-Regularize Unit) ====

def dru(m, sigma=2.0, training=True):
    if training:
        noise = torch.randn_like(m) * sigma
        return torch.sigmoid(m + noise)
    else:
        return (m > 0).float()


def play_full_episode(env, agent1, agent2, optim, gamma, hidden_dim):
    obs = env.reset()
    done = False
    h1 = torch.zeros(1, hidden_dim)
    h2 = torch.zeros(1, hidden_dim)
    m1 = torch.zeros(1, 1)
    m2 = torch.zeros(1, 1)
    r1, r2 = 0.0, 0.0
    agent_1_record = {"obs":[],
                "msg_sent":[],
                "h":[],
                "a":[],
                "r":[],
                "next_obs":[],
                "done": []
                    }

    agent_2_record = {"obs":[],
                    "msg_sent":[],
                    "h":[],
                    "a":[],
                    "r":[],
                    "next_obs":[],
                    "done": []
                    }
    while not done:




        # Agent 1 
        msg2  = dru(m2)
        concat_input_1= torch.cat([torch.Tensor(obs["agent_1"]), msg2], dim =-1)
        q1, m1_next, h1_next = agent1(torch.Tensor(concat_input_1), h1)
  
        
        # epsilon-greedy action selection
        if np.random.rand() < 0.1:
            a1 = np.random.randint(0, env.action_space, 1)[0]
        else:
            a1 = torch.argmax(q1, dim=1).item()

        # Agent 2
        msg1 = dru(m1)
        concat_input_2 = torch.cat([torch.Tensor(obs["agent_2"]), msg1], dim=-1)
        q2, m2_next, h2_next = agent2(torch.Tensor(concat_input_2), h2)
  
        
        # epsilon-greedy action selection
        if np.random.rand() < 0.1:
            a2 = np.random.randint(0, env.action_space, 1)
        else:
            a2 = torch.argmax(q2, dim=1).item()

        # Step
        next_obs, rewards, done = env.step(a1, a2)

        r1 , r2 = rewards
        ## Record
        agent_1_record["obs"].append(obs["agent_1"])
        agent_1_record["msg_sent"].append(m1) #  we are saving the actual message not the encoded one
        agent_1_record["h"].append(h1)
        agent_1_record["a"].append(a1)
        agent_1_record["r"].append(r1)
        agent_1_record["next_obs"].append(next_obs["agent_1"])
        agent_1_record["done"].append(done)
  
        agent_2_record["obs"].append(obs["agent_2"])
        agent_2_record["msg_sent"].append(m2)
        agent_2_record["h"].append(h2)
        agent_2_record["a"].append(a2)
        agent_2_record["r"].append(r2)
        agent_2_record["next_obs"].append(next_obs["agent_2"])
        agent_2_record["done"].append(done)

        
        obs = next_obs
        m1 = m1_next
        m2 = m2_next
        h1 = h1_next
        h2 = h2_next 

    avege_reward = (r1 + r2) / 2
    return agent_1_record, agent_2_record, avege_reward
# ==== Training ====

def apply_dial_algorithm(agent_1_record, agent_2_record, agent1, agent2, optim, gamma, hidden_dim, agnet_1_target):
    loss = 0.0
    ## Reset the gradients of the parameters
    optim.zero_grad()
    gradients_1 = [torch.zeros_like(param) for param in agent1.parameters()]
    gradients_2 = [torch.zeros_like(param) for param in agent2.parameters()]
    gradients_agent = [gradients_1, gradients_2]
    agents = [agent1, agent2]
    history_of_the_game = [agent_1_record, agent_2_record]
    T = len(agent_1_record["done"]) # This is the aount of steps in the game 
    mu_agents_downstream = [[[0.0]]*T,[[0.0]]*T]
    for t in reversed(range(T)):
        for agent_id in range(2):
            agent = agents[agent_id]
            
            agent_history = history_of_the_game[agent_id]
            obs = agent_history["obs"][t]
            next_obs = agent_history["next_obs"][t]
            h = agent_history["h"][t]
            a = agent_history["a"][t]
            r = agent_history["r"][t]

            done = agent_history["done"][t]
            
            other_agetn_m_previuse = history_of_the_game[(agent_id + 1) % 2]["msg_sent"][t]

            if done:
                target = r
            else:
                next_h = agent_history["h"][t+1]
                m2 = history_of_the_game[(agent_id + 1) % 2]["msg_sent"][t+1]
                msg2 = dru(m2)
                input_to_net_t_plus_1 = torch.cat([torch.Tensor(next_obs), msg2], dim=-1)
                q_target,_,_ = agnet_1_target(input_to_net_t_plus_1, next_h)
                target = r + gamma * torch.max(q_target, dim=1)[0].detach()

            dru_a_tag_previuse = dru(other_agetn_m_previuse.detach())
            input_to_net_t = torch.cat([torch.Tensor(obs), dru_a_tag_previuse], dim=-1)
            q_s_a, _, _ = agent(input_to_net_t, h)
            
            del_Q_t_a = target - q_s_a[0][a]

            personal_loss = del_Q_t_a ** 2
            
            personal_grads = torch.autograd.grad(personal_loss, agent.parameters(), retain_graph=True, allow_unused=True)
            
            gradients = gradients_agent[agent_id]
            for i, grad in enumerate(personal_grads):
                if grad is None:
                    continue
                gradients[i] += grad.detach()
            gradients_agent[agent_id] = gradients
            
            loss += personal_loss.item()
            if not done:
                ## calculate the imidiate impact of your message on other agents 
                other_agent_history = history_of_the_game[(agent_id + 1) % 2]
                m_a_t = history_of_the_game[agent_id]["msg_sent"][t].detach().requires_grad_(True)
                other_agent = agents[(agent_id + 1) % 2]
                other_agent_next_observation = other_agent_history["obs"][t+1]
                other_agent_next_h = other_agent_history["h"][t+1].detach()
                other_agent_next_a = other_agent_history["a"][t+1]
                other_agent_next_done  = other_agent_history["done"][t+1]

                dru_m_a_t = dru(m_a_t)
                
                input_to_net_t_plus_1 = torch.cat([torch.Tensor(other_agent_next_observation), dru_m_a_t], dim=-1)
                q_other_agent_t_plus_1,m_a_tag_t_plus_1,_ = other_agent(input_to_net_t_plus_1, other_agent_next_h)
                q_s_a_t_plus_1_other_agent = q_other_agent_t_plus_1[0][other_agent_next_a]

                ## Calculate the target of the otehr agent 
                if other_agent_next_done:
                    target_other_agent_t_plus_1 = other_agent_history["r"][t+1]
                else:
                    other_agent_next_observation_t_plus_2 = other_agent_history["obs"][t+2]
                    other_agent_next_h_t_plus_2 = other_agent_history["h"][t+2].detach()

                    # agent future message impact
                    m_a_t_plus_1 = history_of_the_game[agent_id]["msg_sent"][t+1].detach()
                    dru_m_a_t_plus_1 = dru(m_a_t_plus_1).detach()

                    input_to_net_t_plus_2 = torch.cat([torch.Tensor(other_agent_next_observation_t_plus_2), dru_m_a_t_plus_1], dim=-1)

                    q_target_t_plus_2,_,_ = agnet_1_target(input_to_net_t_plus_2, other_agent_next_h_t_plus_2)    

                    q_s_a_t_plus_2_max = torch.max(q_target_t_plus_2, dim=1)[0].detach()
                    target_other_agent_t_plus_1 = other_agent_history["r"][t+1] + gamma * q_s_a_t_plus_2_max

                del_q_t_plus_1_a_tag = (target_other_agent_t_plus_1 - q_s_a_t_plus_1_other_agent)**2

                d_Q_a_tag_to_m_hat_t_a = torch.autograd.grad(del_q_t_plus_1_a_tag, dru_m_a_t, retain_graph=True, allow_unused=True)

                d_m_a_tag_t_plus_1_d_m_t_hat = torch.autograd.grad(m_a_tag_t_plus_1, dru_m_a_t, retain_graph=True, allow_unused=True)

                mu_t_plus_1_a_tag = mu_agents_downstream[(agent_id + 1) % 2][t+1]
                mu_t_a = d_Q_a_tag_to_m_hat_t_a[0] + mu_t_plus_1_a_tag[0] * d_m_a_tag_t_plus_1_d_m_t_hat[0]

                mu_agents_downstream[agent_id][t] = mu_t_a

                # Now calcuualet the dm_t_a/d_theta

                obs_t_minus_1 = agent_history["obs"][t-1]
                h_t_minus_1 = agent_history["h"][t-1].detach()
                message_other_aget_t_minus_1 = other_agent_history["msg_sent"][t-1].detach()
                message_other_aget_t_minus_1_dru = dru(message_other_aget_t_minus_1).detach()

                t_minus_1_input  = torch.cat([torch.Tensor(obs_t_minus_1), message_other_aget_t_minus_1_dru], dim=-1)
                _,m_t, _ = agent(t_minus_1_input, h_t_minus_1)

                d_m_a_t_to_dtheta = torch.autograd.grad(dru(m_t), agent.parameters(), retain_graph=True, allow_unused=True)

                gradients = gradients_agent[agent_id]
                for i, grad in enumerate(d_m_a_t_to_dtheta):
                    if grad is None:
                        continue
                    gradients[i] += mu_t_a[0] * grad.detach()
                gradients_agent[agent_id] = gradients

    

    
    return loss, gradients_agent


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

    # for param in agent2.parameters():
    #     param.grad = gradint_agent_2[param]

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