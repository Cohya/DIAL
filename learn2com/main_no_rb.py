import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from games.ColorDigitGuessEnv import ColorDigitGuessEnv
from learn2com.agents.lean_to_com_agent import Agent
from learn2com.methods.ddqn import DDQN
from networks.C_Net import C_Net
from learn2com.utils.discretise_regularise_unit import (
    get_discretise_regularise_unit,
)
from learn2com.utils.general import get_learn2com_config
from learn2com.utils.priorized_replay_buffer import PrioritizedReplayBuffer
from learn2com.utils.run_episode import play_one_game
from utils.helpers import smooth_signal


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learn2com_config = get_learn2com_config()
    network_parameters = learn2com_config["network_parameters"]

    C = learn2com_config["taget_update_interval"]
    batch_size = learn2com_config["parallel_episode_batch_size"]
    env = ColorDigitGuessEnv(
        use_mnist=True,
        batch_size=batch_size,
        game_colors=3,
        game_dim=28,
        game_level="extra_hard_local",
        nsteps=2,
        coop=1.0,  # Cooperative environment
    )

    main_net_agent_1 = C_Net(
        obs_dims=env.obs_dims,
        number_of_agents=2,
        action_dims=env.action_space,
        message_dims=1,
        embedding_dim=network_parameters["embedding_dim"],
    )

    main_net_agent_2 = C_Net(
        obs_dims=env.obs_dims,
        number_of_agents=2,
        action_dims=env.action_space,
        message_dims=1,
        embedding_dim=network_parameters["embedding_dim"],
    )

    ####### Target Networks ###
    target_net_agen_1 = C_Net(
        obs_dims=env.obs_dims,
        number_of_agents=2,
        action_dims=env.action_space,
        message_dims=1,
        embedding_dim=network_parameters["embedding_dim"],
    )

    target_net_agen_2 = C_Net(
        obs_dims=env.obs_dims,
        number_of_agents=2,
        action_dims=env.action_space,
        message_dims=1,
        embedding_dim=network_parameters["embedding_dim"],
    )

    ## copy weights from main net
    main_net_agent_2.copy_weights_from_other_network(main_net_agent_1)

    # target nets for DDQN
    target_net_agen_1.copy_weights_from_other_network(main_net_agent_1)
    target_net_agen_2.copy_weights_from_other_network(main_net_agent_1)

    optimizer_conf = learn2com_config["optimizer"]
    learning_rate = optimizer_conf["learning_rate"]
    optimizer_agent_1 = torch.optim.RMSprop(
        params=main_net_agent_1.parameters(),
        lr=learning_rate,
        alpha=0.99,
        momentum=optimizer_conf["momentum"],
    )

    optimizer_agent_2 = torch.optim.RMSprop(
        params=main_net_agent_2.parameters(),
        lr=optimizer_conf["learning_rate"],
        alpha=0.99,
        momentum=optimizer_conf["momentum"],
    )

    gamma = learn2com_config["discount_factor"]

    rl_model_agent_1 = DDQN(
        main_net=main_net_agent_1,
        target_net=target_net_agen_1,
        optimizer=optimizer_agent_1,
        gamma=gamma,
        i_d=1,
    )

    rl_model_agent_2 = DDQN(
        main_net=main_net_agent_2,
        target_net=target_net_agen_2,
        optimizer=optimizer_agent_2,
        gamma=gamma,
        i_d=2,
    )

    replay_buffer_config = learn2com_config["replay_buffer_params"]
    if replay_buffer_config["type"] == "PrioritizedReplayBuffer":
        replay_buffer = PrioritizedReplayBuffer(
            capacity=replay_buffer_config["capacity"],
            alpha=replay_buffer_config["alpha"],
            beta_start=replay_buffer_config["beta_start"],
            beta_frames=replay_buffer_config["beta_frames"],
        )

    agent_1 = Agent(model=rl_model_agent_1, i_d=0, replay_buffer=replay_buffer)
    agent_2 = Agent(model=rl_model_agent_2, i_d=1, replay_buffer=replay_buffer)

    agents = [agent_1, agent_2]

    eps = learn2com_config["espilone"]

    DRU = get_discretise_regularise_unit(learn2com_config["discretise_regularise_unit"])

    number_of_agents = 2

    steps = 0
    average_reward = 0.0
    reward_vec = []
    ratio_reward = []
    for episode in range(5000):
        (
            agent_1_rb,
            agent_2_rb,
            accumulated_rewards,
            T,
            steps,
            accumulated_ratio_rewards,
        ) = play_one_game(
            env=env,
            number_of_agents=number_of_agents,
            batch_size=batch_size,
            agent_1=agent_1,
            agent_2=agent_2,
            network_parameters=network_parameters,
            DRU=DRU,
            eps=eps,
            steps=steps,
        )

        average_reward = average_reward + 1 / (episode + 1) * (
            accumulated_rewards.item() - average_reward
        )

        storage_of_agents_touple = [agent_1_rb, agent_2_rb]
        # print("Now you need to make all gradients :)")
        if (episode + 1) % 50 == 0:
            print(
                f"Episode: {episode}, Average Reward: {average_reward}, ratio_rewards: {np.mean(ratio_reward[-100:])}"
            )

        reward_vec.append(accumulated_rewards.item())
        ratio_reward.append(accumulated_ratio_rewards.item())
        update_weights = True
        if update_weights:
            optimizer = agent_1.model.optimizer
            # collect the gradient
            optimizer.zero_grad()

            mu_t = {agent.i_d: [0] * (T + 1) for agent in agents}

            grad_theta_total = [
                torch.zeros_like(param) for param in agent_1.model.main_net.parameters()
            ]
            for t in range(T - 1, -1, -1):
                y_agents = [None] * number_of_agents

                for i in range(number_of_agents):
                    agent = agents[i]
                    agent_rb = storage_of_agents_touple[i]

                    # Here we should compute oindex_for_t_next, index_to_end
                    index_for_t = t * batch_size
                    index_to_end = index_for_t + batch_size
                    agent_rb_of_t = agent_rb[
                        index_for_t:index_to_end
                    ]  # <---- compute the relevant indexes

                    # if t == T - 1:
                    # Also should be wrt t
                    grad_theta_t = agent.model.get_grad_theta_wrt_q(agent_rb_of_t)
                    # grad_theta_total += grad_theta_t <- wont work since grad_theta_t is a list
                    grad_theta_total = [
                        g1 + g2 for g1, g2 in zip(grad_theta_total, grad_theta_t)
                    ]
                    counter = 0
                    if t < T - 1:
                        for agent_tag in agents:
                            if agent_tag.i_d == agent.i_d:
                                continue

                            # find the impact of the agent message over the loss function of the other agent (agent_tag)
                            agent_tag_rb = storage_of_agents_touple[agent_tag.i_d]
                            # Now it is a little bit tricky , since we are doind batch game and each game has a length of T so we have in
                            # agent_tag_rb len(agent_tag_rb)/T number_of_games

                            index_for_t_next = (t + 1) * batch_size
                            index_to_end = index_for_t_next + batch_size

                            # print(
                            #     f"taking indexes from {index_for_t_next} to {index_to_end}"
                            # )
                            agent_tag_rb_of_t = agent_tag_rb[
                                index_for_t_next:index_to_end
                            ]

                            grad_theta_wrt_message = (
                                agent_tag.model.get_grad_theta_wrt_message(
                                    agent_tag_rb_of_t
                                )
                            )

                            grad_message_t_plus_1_wrt_message, message = (
                                agent_tag.model.get_grad_message_wrt_message(
                                    agent_tag_rb_of_t
                                )
                            )

                            mu_t_p_1_a_tag = mu_t[agent_tag.i_d][t + 1]

                            if counter == 0:
                                summation_of_agent_a_tag = (
                                    grad_theta_wrt_message
                                    + mu_t_p_1_a_tag * grad_message_t_plus_1_wrt_message
                                )

                            else:

                                summation_of_agent_a_tag += (
                                    grad_theta_wrt_message
                                    + mu_t_p_1_a_tag * grad_message_t_plus_1_wrt_message
                                )

                            counter += 1

                        indicator = float(t < (T - 1))
                        mu_t[agent.i_d][t] = indicator * summation_of_agent_a_tag

                        # Here you should take the previous yime step for this calculation
                        index_for_t = (t + 1 - 1) * batch_size
                        index_to_end = index_for_t + batch_size
                        agent_rb_of_t = agent_1_rb[index_for_t:index_to_end]

                        grads_of_dru_wrt_thetas = (
                            agent.model.get_grad_of_dru_mat_wrt_theta(
                                agent_rb_of_t, DRU
                            )
                        )

                        for ij in range(len(grad_theta_total)):
                            if (
                                grads_of_dru_wrt_thetas[ij] is None
                            ):  # This can happens if the input message are nan so reciver layer is inactive
                                continue

                            grad_theta_total[ij] = (
                                grad_theta_total[ij]
                                + mu_t[agent.i_d][t].mean()
                                * grads_of_dru_wrt_thetas[ij]
                                / batch_size
                            )
            # Update the weights

            use_the_optimizer = False
            if use_the_optimizer:  # Use the optimizer:
                for param, grad in zip(
                    agent_1.model.main_net.parameters(), grad_theta_total
                ):
                    param.grad = grad
                optimizer.step()

            else:
                # Do it userelf manually
                for param, grad in zip(
                    agent_1.model.main_net.parameters(), grad_theta_total
                ):
                    param.data = param.data - learning_rate * grad

            ## Update all other agent with the new weights
            for agent in agents:
                if agent.i_d == agent_1.i_d:
                    continue
                agent.model.copy_weights_from_other_network(agent_1.model.main_net)

            # Update the target network interval C
            if steps % C == 0:
                # Copy the weights to the target net
                for agent in agents:
                    agent.model.update_target_net()

    plt.plot(reward_vec, label="Reward")
    plt.plot(smooth_signal(reward_vec, 50), label="Smoothed Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend(frameon=False)
    plt.savefig(
        "reward_plot.png",
        format="png",
    )

    plt.close()
    plt.plot(ratio_reward, label="Ratio Reward")
    plt.plot(smooth_signal(ratio_reward, 50), label="Smoothed Ratio Reward")
    plt.xlabel("Episode")
    plt.ylabel("Ratio Reward")
    plt.legend(frameon=False)
    plt.savefig(
        "ratio_reward_plot.png",
        format="png",
    )


if __name__ == "__main__":
    main()
