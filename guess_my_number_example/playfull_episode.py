
import os
import sys 

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import torch 
import numpy as np 

from guess_my_number_example.discret_regularize_unit import dru
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