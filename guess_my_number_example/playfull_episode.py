
import os
import sys 

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import torch 
import numpy as np 

from guess_my_number_example.discret_regularize_unit import dru
def play_full_episode(env, agent1network, agent2network, hidden_dim, training = True ):
    obs = env.reset()
    done = False

    if agent1network.__class__.__name__ == "AgentNet":
        h1 = torch.zeros(1, hidden_dim)
        h2 = torch.zeros(1, hidden_dim)
    elif agent1network.__class__.__name__ == "C_Net":
        h1 = (torch.zeros(1,1, hidden_dim), torch.zeros(1, 1, hidden_dim))
        h2 = (torch.zeros(1,1, hidden_dim), torch.zeros(1, 1, hidden_dim))
    else:
        raise ValueError(f"Network architecture {agent1network.__class__.__name__} is not supported")
    
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
    
    if training:
        agent1network.train()
        agent2network.train()
    else:
        agent1network.eval()
        agent2network.eval()

    t = 0
    while not done:

        # Agent 1 
        msg2  = dru(m2, training = training)
        if  agent1network.__class__.__name__ == "AgentNet":
            concat_input_1= torch.cat([torch.Tensor(obs["agent_1"]), msg2], dim =-1)
            q1, m1_next, h1_next = agent1network(torch.Tensor(concat_input_1), h1)

        elif  agent1network.__class__.__name__ == "C_Net":
            obs1 = torch.Tensor(obs["agent_1"])
            message = msg2
            u_tm1 = None if t ==0 else agent_1_record["a"][-1]
            h_1 = h1[0]
            h_2 = h1[1]
            a_id  =  torch.tensor(0, dtype=torch.long)
            q1, m1_next, h1_next = agent1network(obs1, message, u_tm1, a_id, h_1, h_2)
        
  
        
        # epsilon-greedy action selection
        if np.random.rand() < 0.1:
            a1 = np.random.randint(0, env.action_space, 1)[0]
        else:
            a1 = torch.argmax(q1, dim=1).item()

        # Agent 2
        msg1 = dru(m1)

        if agent2network.__class__.__name__ == "AgentNet":
            concat_input_2 = torch.cat([torch.Tensor(obs["agent_2"]), msg1], dim=-1)
            q2, m2_next, h2_next = agent2network(torch.Tensor(concat_input_2), h2)

        elif agent2network.__class__.__name__ == "C_Net":
            obs2 = torch.Tensor(obs["agent_2"])
            message = msg1
            u_tm1 = None if t ==0 else agent_2_record["a"][-1]
            a_id =  torch.tensor(1, dtype=torch.long)
            h_1 = h2[0]
            h_2 = h2[1]
            q2, m2_next, h2_next = agent2network(obs2, message, u_tm1, a_id, h_1, h_2)
  
        
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
        agent_1_record["msg_sent"].append(m1.detach()) #  we are saving the actual message not the encoded one
        if agent1network.__class__.__name__ == "AgentNet":
            agent_1_record["h"].append(h1.detach())
        else:
            agent_1_record["h"].append([hi.detach() for hi in h1])

        agent_1_record["a"].append(a1)
        agent_1_record["r"].append(r1)
        agent_1_record["next_obs"].append(next_obs["agent_1"])
        agent_1_record["done"].append(done)

  
        agent_2_record["obs"].append(obs["agent_2"])
        agent_2_record["msg_sent"].append(m2.detach())
        if agent1network.__class__.__name__ == "AgentNet":
            agent_2_record["h"].append(h2.detach())
        else:
            agent_2_record["h"].append([hi.detach() for hi in h2])

        agent_2_record["a"].append(a2)
        agent_2_record["r"].append(r2)
        agent_2_record["next_obs"].append(next_obs["agent_2"])
        agent_2_record["done"].append(done)

        
        obs = next_obs
        m1 = m1_next
        m2 = m2_next
        h1 = h1_next
        h2 = h2_next 

        t += 1

    avege_reward = (r1 + r2) / 2
    return agent_1_record, agent_2_record, avege_reward