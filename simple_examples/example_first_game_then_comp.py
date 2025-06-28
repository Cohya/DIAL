import os 
import sys 
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RMSprop
import random
from simple_examples.nets import MessageNetRecuurent, QNet
from simple_examples.guess_my_number import GuessMyNumberEnv
import numpy as np 
from learn2com.utils.discretise_regularise_unit import discretise_regularise_unit
import matplotlib.pyplot as plt
from  learn2com.nn_networks.C_Net import C_Net
def dial_episode(i, batch_size = 1):

    obs1, obs2 = env.reset()
    total_reward1, total_reward2 = 0.0, 0.0
    number_of_agents = 2
    history = []  # to store everything per step for backward pass
    # msg2, msg1 = torch.tensor(float("nan")).unsqueeze(0).unsqueeze(0), torch.tensor(float("nan")).unsqueeze(0).unsqueeze(0)
    messages_history = torch.Tensor(
        [[float("nan"), float("nan")]] * batch_size
    ).unsqueeze(
        axis=1
    )  # At the beginning there is no message (B, 1, A)
    msg2, msg1 = messages_history[:,:,0], messages_history[:,:,1]
    h1_1, h2_1 = torch.zeros(1,batch_size, 128) , torch.zeros(1,batch_size, 128) # GRU cell  inside 
    h1_2, h2_2 = torch.zeros(1,batch_size, 128) , torch.zeros(1,batch_size, 128) # GRU cell  inside 
    # a1, a2 = torch.tensor(float("nan")).unsqueeze(0), torch.tensor(float("nan")).unsqueeze(0)

    actions_tm1 = torch.Tensor([[float("nan"), float("nan")]] * batch_size).unsqueeze(
        axis=1
    )  # At the beginning there is no action (B, 1, A)
    a1, a2 = actions_tm1[:,:,0], actions_tm1[:,:,1]

    r1, r2 = torch.stack([torch.tensor(0.0)]*batch_size,dim = -1).unsqueeze(1),torch.stack([torch.tensor(0.0)]*batch_size,dim = -1).unsqueeze(1)
    agent_id_1 = 0
    agent_id_2 = 1
    for step in range(env.max_steps):
 
        msg2_dru = discretise_regularise_unit(msg2, scale=0.1, training=True) if msg2 is not None else None
        msg1_dru = discretise_regularise_unit(msg1, scale=0.1, training=True) if msg1 is not None else None
    
        q1, msg1_next, out1, h1_1_next, h1_2_next = cnet(obs1, msg2_dru,a1, agent_id_1 , h1_1, h2_1) # <- must be recurretn 
        q2, msg2_next, out2, h2_1_next, h2_2_next = cnet(obs2, msg1_dru,a2, agent_id_2 , h1_2, h2_2) 

        ## This we should do in loop 

        # Generate random numbers for each batch element
        random_vals = torch.rand(batch_size, device=q1.device)
        # Mask for greedy (argmax) selection
        greedy_mask = random_vals > 0.05

        # Argmax actions
        greedy_actions = q1.argmax(dim=-1)

        # Random actions
        random_actions = torch.randint(low=0, high=10, size=(batch_size,), device=q1.device)

        # Combine using the mask
        a1 = torch.where(greedy_mask, greedy_actions, random_actions)

        random_vals_2 = torch.rand(batch_size, device=q2.device)
        greedy_mask_2 = random_vals_2 > 0.05
        greedy_actions_2 = q2.argmax(dim=-1)
        random_actions_2 = torch.randint(low=0, high=10, size=(batch_size,), device=q2.device)
        a2 = torch.where(greedy_mask_2, greedy_actions_2, random_actions_2)

        ####
        # a1 = q1[0].argmax(dim=-1).unsqueeze(0) if  np.random.random() > 0.1 else  torch.randint(low=0, high=10, size=(1,))
        # a2 = q2[0].argmax(dim=-1).unsqueeze(0) if  np.random.random() > 0.1 else  torch.randint(low=0, high=10, size=(1,))

        obs_next, rewards, done = env.step(a1, a2)
        obs1_next, obs2_next = obs_next
        r1, r2 = rewards
        total_reward1 += torch.mean(r1).item()
        total_reward2 += torch.mean(r2).item()

        h = {
            'obs1': obs1.detach().clone(),
            'obs2': obs2.detach().clone(),
            'msg1': msg1.detach().clone(),
            'msg2': msg2.detach().clone(),
            'h1_1': h1_1.detach().clone(),
            'h2_1': h2_1.detach().clone(),
            'h1_2': h1_2.detach().clone(),
            'h2_2': h2_2.detach().clone(),
            'a1': a1.detach().clone(),
            'a2': a2.detach().clone(),
            'r1': r1,
            'r2': r2,
            't': step
        }
        h['obs1_next'] = obs1_next.detach().clone()
        h['obs2_next'] = obs2_next.detach().clone()
        h['msg1_next'] = msg1_next.detach().clone()
        h['msg2_next'] = msg2_next.detach().clone()

        h['h1_1_next'] = h1_1_next.detach().clone()
        h['h1_2_next'] = h1_2_next.detach().clone()

        h['h2_2_next'] = h2_2_next.detach().clone()
        h['h2_1_next'] = h2_1_next.detach().clone()

        h['a1_next'] = a1.detach().clone()
        h['a2_next'] = a2.detach().clone()
        h['done'] = done
        history.append(h)

        obs1, obs2 = obs1_next, obs2_next
        msg1, msg2 = msg1_next, msg2_next
        h1_1, h2_1 = h1_1_next, h2_1_next
        h1_2, h2_2 = h1_2_next, h2_2_next
        
        if done:
            break
            

   
    # Step 1: backward Q loss for all steps
    # total_loss = sum([step['q_loss'] for step in history])
    # total_loss.backward(retain_graph=True)

    # Step 2: backward through messages (recursive from T to 0)
    mu1, mu2 = torch.tensor(0.0), torch.tensor(0.0)
    N = len(history)
    theta_grads = [torch.zeros_like(param) for param in cnet.parameters()]
    total_loss_to_return = 0
    for t in range(N-1, 0, -1):
        # print(t)
        step = history[t]
        
        obs1 = step['obs1']
        obs1_next = step['obs1_next']
        msg1 = step['msg1']
        msg1_next = step['msg1_next']
        h1_1 = step['h1_1']
        h1_2 = step['h1_2']
        h1_1_next = step['h1_1_next']
        h1_2_next = step['h1_2_next']
        a1 = step['a1']
        r1 = step['r1']
        a1_next = step['a1_next']

        obs2 = step['obs2']
        obs2_next = step['obs2_next']
        msg2 = step['msg2']
        msg2_next = step['msg2_next']
        a2 = step['a2']
        h2_1 = step['h2_1']
        h2_1_next = step['h2_1_next']
        h2_2 = step['h2_2']
        h2_2_next = step['h2_2_next']
        r2 = step['r2']
        a2_next = step['a2_next']
        
        t = step['t']
        done = step['done']
 
        with torch.no_grad():
            if done :#t == env.max_steps - 1:
    
                target_1 = r1 if isinstance(r1, torch.Tensor) else torch.tensor([r1])
                target_2 = r2 if isinstance(r2, torch.Tensor) else torch.tensor([r2])

            else:
                #obs2.float(), msg1_dru,a2, agent_id_2 , h1_2, h2_2
                ## This is DQN not DDQN !! you should fix 
                msg1_next_after_dru = discretise_regularise_unit(msg1_next, scale=0.1, training=True)
                msg2__next_after_dru = discretise_regularise_unit(msg2_next, scale=0.1, training=True)
                q1, _, _, _, _ = cnet_target(obs1_next, msg2__next_after_dru,a1_next, agent_id_1, h1_1_next,h1_2_next)
                q1  = q1.max(dim=1).values
                target_1 = r1 + gamma * q1

                q2, _, _, _, _ =  cnet_target(obs2_next, msg1_next_after_dru,a2_next, agent_id_2, h2_1_next,h2_2_next)
                q2 = q2.max(dim=1).values
                target_2 = r2 + gamma * q2

        ## Extract t - 1

        ## My 
        msg1_t = msg1.clone().detach().requires_grad_(True)
        msg2_t = msg2.clone().detach().requires_grad_(True)


        msg1_after_dru = discretise_regularise_unit(msg1_t, scale=0.1, training=True)
        msg2_after_dru = discretise_regularise_unit(msg2_t, scale=0.1, training=True)


        
        q_value_1_estimated,msg_1_nextt,_,_,_ = cnet(obs1, msg2_after_dru,a1, agent_id_1, h1_1, h1_2)#.gather(1, a1.unsqueeze(1)).squeeze(1)
        q_value_2_estimated,msg_2_nextt,_,_,_= cnet(obs2, msg1_after_dru,a2, agent_id_2, h2_1, h2_2)#.gather(1, a2.unsqueeze(1)).squeeze(1)

        action_value_1 = q_value_1_estimated.gather(1, a1.unsqueeze(1)).squeeze(1)
        action_value_2 = q_value_2_estimated.gather(1, a2.unsqueeze(1)).squeeze(1)

        td_error_1 = (target_1 - action_value_1)
        td_error_2 = (target_2 - action_value_2)

        mse_q1_loss =td_error_1.pow(2).mean()
        mse_q2_loss =td_error_2.pow(2).mean()
        # print("mse_q2_loss:" , mse_q2_loss)
        total_loss = mse_q1_loss + mse_q2_loss
        total_loss_to_return += total_loss.detach().clone()
        # Backward pass for Q loss
        td_error_gradients  = torch.autograd.grad(
            outputs=total_loss,
            inputs=list(cnet.parameters()),
            retain_graph=True,
            create_graph=True
        )

        ## add like in the article 
        for i in range(len(td_error_gradients)):
            theta_grads[i] += td_error_gradients[i]

        
        ## Now  we need to check the impact of the messages  over the q of other agents 
        ## Lets coimpute the impact of m(t) on message(t+1)
        # ... use msg1 in computation that leads to mse_q2_loss ...

        d_Q_a_tag_to_msg_a_1 = torch.autograd.grad(outputs=mse_q2_loss, inputs=msg1_t, retain_graph=True, create_graph=True)

        d_m_tplus_1_to_msg_a_aggregate = torch.zeros_like(msg1_t)
        for mkj in range(batch_size): 
            d_m_tplus_1_to_msg_a = torch.autograd.grad(outputs=msg_2_nextt[mkj], inputs=msg1_t, retain_graph=True, create_graph=True, allow_unused=True)
            d_m_tplus_1_to_msg_a_aggregate += d_m_tplus_1_to_msg_a[0]/batch_size

        d_m_tplus_1_to_msg_a = (d_m_tplus_1_to_msg_a_aggregate,)



        d_Q_a_tag_to_msg_a_2 = torch.autograd.grad(outputs=mse_q1_loss, inputs=msg2_t, retain_graph=True, create_graph=True)

        d_m_tplus_1_to_msg_a_aggregate_2 = torch.zeros_like(msg2_t)
        for mkj in range(batch_size): 
            d_m_tplus_1_to_msg_a_2 = torch.autograd.grad(outputs=msg_1_nextt[mkj], inputs=msg2_t, retain_graph=True, create_graph=True, allow_unused=True)
            d_m_tplus_1_to_msg_a_aggregate_2 += d_m_tplus_1_to_msg_a_2[0]/batch_size

        d_m_tplus_1_to_msg_a_2 = (d_m_tplus_1_to_msg_a_aggregate_2,)


        mu1 = float(t < (N-1)) * d_Q_a_tag_to_msg_a_1[0].detach() + mu1.detach() * d_m_tplus_1_to_msg_a[0].detach() #  (batch_size,1)
        mu2 = float(t < (N-1)) * d_Q_a_tag_to_msg_a_2[0].detach() + mu2.detach() * d_m_tplus_1_to_msg_a_2[0].detach()

        step_t_minus_1 = history[t-1]
        # print("t: ", t, "mu1", mu1, "mu2", mu2)
        msg_2_t_minus_1 = step_t_minus_1["msg2"]
        obs1_t_minus_1 = step_t_minus_1["obs1"]
        a1_t_minus_1 = step_t_minus_1["a1"]
        h1_1_t_minus_1 = step_t_minus_1["h1_1"]
        h1_2_t_minus_1 = step_t_minus_1["h1_2"]


      
        msg_1_t_minus_1 = step_t_minus_1["msg1"]
        obs2_t_minus_1 = step_t_minus_1["obs2"]
        a2_t_minus_1 = step_t_minus_1["a2"]
        h2_1_t_minus_1 = step_t_minus_1["h2_1"]
        h2_2_t_minus_1 = step_t_minus_1["h2_2"]

        
        _,message_1, _,_,_ = cnet(obs1_t_minus_1, msg_2_t_minus_1,a1_t_minus_1, agent_id_1, h1_1_t_minus_1, h1_2_t_minus_1)
        message_1_dru = discretise_regularise_unit(message_1, scale=0.1, training=True)
        # param_names = [name for name, _ in cnet.named_parameters()]

        aggregate_theta_grad_wrt_message = [torch.zeros_like(param) for param in cnet.parameters()]
        for jhg in range(batch_size):
            derivative_wrt_theta_agent_1 = torch.autograd.grad(
                outputs=message_1_dru[jhg],
                inputs=list(cnet.parameters()),
                retain_graph=True,
                create_graph=True,
                allow_unused=True
            )   
            # here I think we should devide by batch_size
            for jhg2 in range(len(aggregate_theta_grad_wrt_message)):
                if  derivative_wrt_theta_agent_1[jhg2] is not None:
                  aggregate_theta_grad_wrt_message[jhg2] += derivative_wrt_theta_agent_1[jhg2]/batch_size

                # else:
                #     print(param_names[jhg2])
                #     print("t:", t)
                #     print("derivative_wrt_theta_agent_1[jhg2] is None")
                    
            

        _,message_2, _,_,_ = cnet(obs2_t_minus_1, msg_1_t_minus_1,a2_t_minus_1, agent_id_2, h2_1_t_minus_1, h2_2_t_minus_1)
        message_2_dru = discretise_regularise_unit(message_2, scale=0.1, training=True)
        
        aggregane_theta_grad_wrt_message_2 = [torch.zeros_like(param) for param in cnet.parameters()]
        for jhg in range(batch_size):
            derivative_wrt_theta_2_agent_2 = torch.autograd.grad(
                outputs=message_2_dru[jhg],
                inputs=list(cnet.parameters()),
                retain_graph=True,
                create_graph=True,
                allow_unused = True
            )

            # here I think we should devide by batch_size
            for jhg2 in range(len(aggregane_theta_grad_wrt_message_2)):
                if  derivative_wrt_theta_2_agent_2[jhg2] is not None:
                  aggregane_theta_grad_wrt_message_2[jhg2] += derivative_wrt_theta_2_agent_2[jhg2]/batch_size


        derivative_wrt_theta_agent_1 = aggregate_theta_grad_wrt_message
        derivative_wrt_theta_2_agent_2 = aggregane_theta_grad_wrt_message_2

        for jk in range(len(theta_grads)):
            if  derivative_wrt_theta_agent_1[jk] is not None:
                 theta_grads[jk] += mu1.mean()* derivative_wrt_theta_agent_1[jk] #was mu1[0]
            if  derivative_wrt_theta_2_agent_2[jk] is not None:
                 theta_grads[jk] += mu2.mean()* derivative_wrt_theta_2_agent_2[jk]

    optimizer.zero_grad(set_to_none=True)
    for param, grad in zip(cnet.parameters(), theta_grads):
        param.grad = grad/number_of_agents
    optimizer.step()
    return total_loss_to_return.item(), total_reward1, total_reward2


# Run training
all_exp = dict()
batch_size = 16
env = GuessMyNumberEnv(batch_size=batch_size, max_steps=5)
cnet = C_Net(obs_dims=[1], number_of_agents=2, action_dims=10, message_dims = 1, embedding_dim=128)
cnet_target = C_Net(obs_dims=[1], number_of_agents=2, action_dims=10,message_dims = 1, embedding_dim=128)


optimizer = RMSprop(
    list(cnet.parameters()), lr=5e-4, momentum=0.95
)
optimizer.zero_grad()
gamma = 1.0
loss_vec, reward_vec = [], []
aveg_r, aveg_loss = [], []

fig, ax = plt.subplots()
for i in range(5000):
    if i % 100 == 0:
        cnet_target.copy_weights_from_other_network(cnet)


    loss, reward1, reward2 = dial_episode(i, batch_size=batch_size)
    reward_vec.append((reward1 + reward2) / 2)
    loss_vec.append(loss)
    if i % 100 == 0:
        print(f"Running episode {i}...")
        print(f"Average Reward: {np.mean(reward_vec[-100:]):.4f}, Loss: {np.mean(loss_vec[-100:]):.8f}")


    aveg_r.append(np.mean(reward_vec[-100:]))
    aveg_loss.append(np.mean(loss_vec[-100:]))

    if i % 200 == 0:
        ax.plot(aveg_r)
        ax.plot(aveg_loss)
        ax.legend(["Average Reward", "Average Loss"])
        ax.set_title("Average Reward and Loss")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Reward and Loss")
        plt.savefig("average_reward_loss.png")