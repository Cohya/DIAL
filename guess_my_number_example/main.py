import os 
import sys 

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from networks.simple_network import AgentNet, AgentNet2
from networks.C_Net import C_Net
from guess_my_number_example.guess_my_number import GuessMyNumberEnv
import matplotlib.pyplot as plt 
import torch
import numpy as np
import pickle 
from guess_my_number_example.playfull_episode import play_full_episode
from guess_my_number_example.discret_regularize_unit import dru
from guess_my_number_example.dial_algorithm import apply_dial_algorithm
from pathlib import Path  
from typing import Dict
from utils.load_ymal import load_yaml
from utils.get_folder_for_experiment import get_folder_for_experiment


def main(config: Dict):
    input_dim = config["input_dim"]
    hidden_dim = config["hidden_dim"]
    msg_dim = config["msg_dim"]
    action_dim = config["action_dim"]
    max_steps = config["max_steps"] #num of bits are 5-1
    number_of_agents = config["number_of_agents"]
    netwotk_architecture = config["netwotk_architecture"] # ""C_Net" # "simple_network" #"C_Net"
    batch_size = config["batch_size"]
    
    number_of_max_episodes = config["number_of_max_episodes"]
    update_target = config["update_target"] # Update target network every 100 episodes
    print_interval = config["print_interval"] # Print every 1000 episodes
    check_your_performance_interval = config["check_your_performance_interval"] # Check performance every N episodes
    ## Cretae folder for saving the experiment 
    path_to_save = get_folder_for_experiment(config)
    checkpoint_folder = path_to_save / "checkpoints"
    inference_performance = []
    if os.path.isdir(checkpoint_folder) is False:
        os.makedirs(checkpoint_folder)


    if netwotk_architecture == "AgentNet":
        agent1network = AgentNet(input_dim + msg_dim, hidden_dim, msg_dim, action_dim)
        agent2network = AgentNet(input_dim + msg_dim, hidden_dim, msg_dim, action_dim)
        agnet_1_target = AgentNet(input_dim + msg_dim, hidden_dim, msg_dim, action_dim)
        
    elif netwotk_architecture == "C_Net":
        agent1network = C_Net(obs_dims=input_dim, number_of_agents=number_of_agents, action_dims=action_dim, message_dims=msg_dim, embedding_dim=hidden_dim)
        agent2network = C_Net(obs_dims=input_dim, number_of_agents=number_of_agents, action_dims=action_dim, message_dims=msg_dim, embedding_dim=hidden_dim)
        agnet_1_target = C_Net(obs_dims=input_dim, number_of_agents=number_of_agents, action_dims=action_dim, message_dims=msg_dim, embedding_dim=hidden_dim)

    elif netwotk_architecture == "AgentNet2":
        agent1network = AgentNet2(input_dim, hidden_dim, msg_dim, action_dim)
        agent2network = AgentNet2(input_dim, hidden_dim, msg_dim, action_dim)
        agnet_1_target = AgentNet2(input_dim, hidden_dim, msg_dim, action_dim)
    else:
        raise ValueError(f"Network architecture {netwotk_architecture} is not supported")
    # Share weights 
    agent2network.load_state_dict(agent1network.state_dict())
    agnet_1_target.load_state_dict(agent1network.state_dict())
    optim = torch.optim.Adam(list(agent1network.parameters()) + list(agent2network.parameters()), lr=config["learning_rate"])#was 3

    env = GuessMyNumberEnv(max_steps = max_steps, action_space = action_dim)
    gamma = config["gamma"] #0.9
    loss_vec = []
    average_r = []
    max_infernec_avege_reward = -sys.maxsize
    optim.zero_grad()
    batch_gradient_of_param = [torch.zeros_like(param) for param in agent1network.parameters()]

    for episode in range(number_of_max_episodes):

        agent_1_record, agent_2_record , avege_reward = play_full_episode(env, agent1network, agent2network, hidden_dim, training=True, epsilon=config["epsilon"])

        average_r.append(avege_reward)

        loss,gradients_agent = apply_dial_algorithm(agent_1_record, agent_2_record, agent1network, agent2network, optim, gamma, agnet_1_target)

        loss_vec.append(loss)
        ## lets applay the gradient to the network 
        gradint_agent_1 = gradients_agent[0]
        gradint_agent_2 = gradients_agent[1]
        

        for grad, grad2, batch_gradient_of_param_i in zip(gradint_agent_1, gradint_agent_2, batch_gradient_of_param):
            gradient_of_param = (grad + grad2) / 2
            if gradient_of_param is None:
                continue
            batch_gradient_of_param_i += gradient_of_param
            # param.grad = gradient_of_param

        if (episode+1) % batch_size == 0:
            for param, batch_gradient_of_param_i in zip(agent1network.parameters(), batch_gradient_of_param):
                if gradient_of_param is None:
                    continue
                param.grad = batch_gradient_of_param_i /  batch_size
            optim.step()
            optim.zero_grad()
            batch_gradient_of_param = [torch.zeros_like(param) for param in agent1network.parameters()]

        # copy weights
        agent2network.load_state_dict(agent1network.state_dict())
        if episode % update_target == 0:
            agnet_1_target.load_state_dict(agent1network.state_dict())

        if episode % print_interval == 0:
            print("episode: ", episode, "average reward: ", np.mean(average_r[-100:]), "loss: ", np.mean(loss_vec[-100:]))

        if episode % check_your_performance_interval == 0:
            avege_reward_infernece = 0
            for _ in range(100):
                agent_1_record, agent_2_record , avege_reward =  play_full_episode(env, agent1network, agent2network, hidden_dim, training=False)
                avege_reward_infernece += avege_reward

            if avege_reward == 1:
                print("Agent 1 obs:", agent_1_record["obs"][-1], "agent_1_messages:", [int(dru(me.detach(), training=False).detach().numpy()[0][0]) for me in agent_1_record["msg_sent"]])
                print("Agent 2 obs:", agent_2_record["obs"][-1], "agent_2_messages:", [int(dru(me.detach(), training=False).detach().numpy()[0][0]) for me in agent_2_record["msg_sent"]])
            
            
            avege_reward_infernece /= 100
            inference_performance.append((episode, avege_reward_infernece))
            if avege_reward_infernece > max_infernec_avege_reward:
                max_infernec_avege_reward = avege_reward_infernece
                path_to_sv =  checkpoint_folder  / f"agent1_{netwotk_architecture}_best_inference_iteration_{episode}.pth"

                path_to_save_agent_2 =  checkpoint_folder / f"agent2_{netwotk_architecture}_best_inference_iteration_{episode}.pth"
                torch.save(agent1network.state_dict(),path_to_sv)
                torch.save(agent2network.state_dict(), path_to_save_agent_2)
                print("Saved model with max average reward: ", max_infernec_avege_reward)

            

    with open(path_to_save /"loss_vec.pk", "wb") as file:
        pickle.dump(loss_vec, file)

    with open(path_to_save/"average_r.pk", "wb") as file:
        pickle.dump(average_r,file)


    # smooth the averag_r and create a plot 

    average_r = np.array(average_r)
    average_r = np.convolve(average_r, np.ones(100)/100, mode='valid')
    plt.plot(average_r, label = "Average Reward training")
    
    x_inference = [x[0] for x in inference_performance]
    y_inference = [x[1] for x in inference_performance]
    plt.plot(x_inference, y_inference, label = "Average Reward inference")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend(frameon = False)
    #save the iage 
    plt.savefig(path_to_save/"average_r.png")
    plt.close()
    # cerate the same for the loss_vec 
    loss_vec = np.array(loss_vec)
    loss_vec = np.convolve(loss_vec, np.ones(100)/100, mode='valid')
    plt.plot(loss_vec)
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.savefig(path_to_save/ "loss_vec.png")

if __name__ == "__main__":
    path_to_config = Path("guess_my_number_example") / "config.yaml"
    config = load_yaml(path_to_config)
    main(config = config)