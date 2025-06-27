import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from typing import Callable

import torch


def play_one_game(
    env: object,
    number_of_agents: int,
    batch_size: int,
    agent_1: object,
    agent_2: object,
    network_parameters: dict,
    DRU: Callable,
    eps: float,
    steps: int,
):
    messages_history = torch.Tensor(
        [[float("nan"), float("nan")]] * batch_size
    ).unsqueeze(
        axis=1
    )  # At the beginning there is no message (B, 1, A)

    messages_for_next_step = torch.Tensor(
        [[float("nan"), float("nan")]] * batch_size
    ).unsqueeze(
        axis=1
    )  # [B, 1, A]

    actions_tm1 = torch.Tensor([[float("nan"), float("nan")]] * batch_size).unsqueeze(
        axis=1
    )  # At the beginning there is no action (B, 1, A)

    gru_h = [
        [
            torch.zeros(1, batch_size, network_parameters["embedding_dim"]),
            torch.zeros(1, batch_size, network_parameters["embedding_dim"]),
        ]
    ] * number_of_agents  # should be with the shape (1, B, embedded_dim)
    states_for_forward_pass = [None] * number_of_agents
    agents = [agent_1, agent_2]
    agent_2_rb = []
    agent_1_rb = []
    observations = env.reset()
    done = False
    # @TODO(Yaniv): there is abatch of games so you need to consider it when saving into replaybuffer and when you perform done condition!!
    T = 0
    accumulated_rewards = 0
    accumulated_ratio_rewards = 0
    while not done:
        steps += 1
        actions = []
        T += 1
        for i in range(number_of_agents):
            obs = observations[i]
            agent = agents[i]
            message = messages_history[:, :, (i + 1) % 2]
            u_tm1 = actions_tm1[:, 0, i]  # this is the previouse action of the agent
            a = [i] * batch_size  # this is the agent id
            h_1 = gru_h[i][0]
            h_2 = gru_h[i][1]
            h_1 = h_1.detach().clone()
            h_2 = h_2.detach().clone()

            states_for_forward_pass[i] = (obs, message, u_tm1, a, h_1, h_2)

            action, message, out2, h1, h2 = agent.get_action_and_message(
                eps=eps,
                obs=obs,
                message=message,
                u_tm1=u_tm1,
                a=a,
                h_1=h_1,
                h_2=h_2,
            )

            message = DRU(
                message, training=True
            )  # Discretise and regularise the message
            actions.append(action)

            gru_h[i][0] = h1.detach().clone()
            gru_h[i][1] = h2.detach().clone()
            actions_tm1[:, :, i] = action.unsqueeze(axis=1)
            messages_for_next_step[:, :, i] = message.detach().clone()

        # actions_torch = torch.stack(actions, dim=1)
        # Create a  copy of the messages for the future step
        messages_history = messages_for_next_step.clone()

        next_observations, rewards, dones, info = env.step(actions)
        max_reward = info["max_reward"]
        accumulated_rewards += rewards.sum() / batch_size / number_of_agents

        # TODO(Yaniv): you should  think about it again 1!!!
        try:
            # ratio = torch.abs(max_reward - rewards) / (torch.abs(max_reward) + 0.0001)
            ratio = torch.abs(max_reward - rewards).mean()
        except:
            ratio = torch.zeros_like(rewards)

        # accumulated_ratio_rewards += (
        #     torch.abs(ratio).sum() / batch_size / number_of_agents
        # )
        accumulated_ratio_rewards += ratio
        state_1 = states_for_forward_pass[0]
        state_2 = states_for_forward_pass[1]

        next_states = [None] * number_of_agents
        for i in range(number_of_agents):

            next_obs = next_observations[i]
            next_message = messages_history[:, :, (i + 1) % 2]
            u_tm1 = actions_tm1[:, 0, i]  # this is the previouse action of the agent
            a = i  # this is the agent id
            h_1 = gru_h[i][0]
            h_2 = gru_h[i][1]

            next_states[i] = (next_obs, next_message, u_tm1, a, h_1, h_2)

        observations = next_observations

        ## Here you should take into account the number of batches
        for b in range(batch_size):
            ## For  Agent 1

            state = (
                state_1[0][b].unsqueeze(0),  # obs
                state_1[1][b],  # message
                state_1[2][b],  # u_tm1
                state_1[3][b],  # a (agent id, not batched)
                (
                    state_1[4][0:1, b, :]
                    if state_1[4] is not None
                    else torch.zeros(1, batch_size, network_parameters["embedding_dim"])
                ),  # h_1
                (
                    state_1[5][0:1, b, :]
                    if state_1[5] is not None
                    else torch.zeros(1, batch_size, network_parameters["embedding_dim"])
                ),  # h_2
            )
            action = actions[0][b]
            reward = rewards[b][0]
            next_state = (
                next_states[0][0][b].unsqueeze(0),
                next_states[0][1][b],
                next_states[0][2][b],
                next_states[0][3],
                (
                    next_states[0][4][0:1, b, :]
                    if next_states[0][4] is not None
                    else None
                ),
                (
                    next_states[0][5][0:1, b, :]
                    if next_states[0][5] is not None
                    else None
                ),
            )
            done = dones[b]

            agent_1_rb.append((state, action, reward, next_state, done))

            # For agent 2

            state = (
                state_2[0][b].unsqueeze(0),
                state_2[1][b],
                state_2[2][b],
                state_2[3][b],
                state_2[4][0:1, b, :] if state_2[4] is not None else None,
                state_2[5][0:1, b, :] if state_2[5] is not None else None,
            )
            action = actions[1][b]
            reward = rewards[b][1]
            next_state = (
                next_states[1][0][b].unsqueeze(0),
                next_states[1][1][b],
                next_states[1][2][b],
                next_states[1][3],
                (
                    next_states[1][4][0:1, b, :]
                    if next_states[1][4] is not None
                    else None
                ),
                (
                    next_states[1][5][0:1, b, :]
                    if next_states[1][5] is not None
                    else None
                ),
            )
            done = dones[0]

            agent_2_rb.append((state, action, reward, next_state, done))

    return (
        agent_1_rb,
        agent_2_rb,
        accumulated_rewards,
        T,
        steps,
        accumulated_ratio_rewards,
    )
