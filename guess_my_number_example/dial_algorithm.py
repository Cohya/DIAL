import  os 
import sys 
if os.getcwd() not  in sys.path:
    sys.path.append(os.getcwd())

import torch 

from guess_my_number_example.discret_regularize_unit import  dru 

def apply_dial_algorithm(agent_1_record, agent_2_record, agent1, agent2, optim, gamma, agnet_1_target):
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