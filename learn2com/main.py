import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import torch

from nn_models.games.ColorDigitGuessEnv import ColorDigitGuessEnv
from nn_models.learn2com.agents.lean_to_com_agent import Agent
from nn_models.learn2com.methods.ddqn import DDQN
from nn_models.learn2com.nn_networks.C_Net import C_Net
from nn_models.learn2com.utils.discretise_regularise_unit import (
    get_discretise_regularise_unit,
)
from nn_models.learn2com.utils.general import get_learn2com_config
from nn_models.learn2com.utils.priorized_replay_buffer import PrioritizedReplayBuffer


def main():

    learn2com_config = get_learn2com_config()
    network_parameters = learn2com_config["network_parameters"]

    batch_size = learn2com_config["parallel_episode_batch_size"]
    env = ColorDigitGuessEnv(
        use_mnist=True,
        batch_size=batch_size,
        game_colors=3,
        game_dim=28,
        game_level="extra_hard_local",
        nsteps=2,
        coop=0.0,
    )

    main_net = C_Net(
        obs_dims=env.obs_dims,
        number_of_agents=2,
        action_dims=env.action_space,
        message_dims=1,
        embedding_dim=network_parameters["embedding_dim"],
    )

    target_net = C_Net(
        obs_dims=env.obs_dims,
        number_of_agents=2,
        action_dims=env.action_space,
        message_dims=1,
        embedding_dim=network_parameters["embedding_dim"],
    )
    optimizer_conf = learn2com_config["optimizer"]

    optimizer = torch.optim.RMSprop(
        params=main_net.parameters(),
        lr=optimizer_conf["learning_rate"],
        alpha=0.99,
        momentum=optimizer_conf["momentum"],
    )

    rl_model = DDQN(
        main_net=main_net,
        target_net=target_net,
        optimizer=optimizer,
        gamma=learn2com_config["discount_factor"],
    )

    replay_buffer_config = learn2com_config["replay_buffer_params"]
    if replay_buffer_config["type"] == "PrioritizedReplayBuffer":
        replay_buffer = PrioritizedReplayBuffer(
            capacity=replay_buffer_config["capacity"],
            alpha=replay_buffer_config["alpha"],
            beta_start=replay_buffer_config["beta_start"],
            beta_frames=replay_buffer_config["beta_frames"],
        )

    agent_1 = Agent(model=rl_model, i_d=0, replay_buffer=replay_buffer)
    agent_2 = Agent(model=rl_model, i_d=1, replay_buffer=replay_buffer)
    eps = learn2com_config["espilone"]

    DRU = get_discretise_regularise_unit(learn2com_config["discretise_regularise_unit"])
    number_of_agents = 2
    messages = torch.Tensor(
        [[float("nan"), float("nan")]] * batch_size
    )  # At the beginning there is no message
    actions_tm1 = torch.Tensor(
        [[float("nan"), float("nan")]] * batch_size
    )  # At the beginning there is no action
    # [
    #     [None, None]
    # ] * number_of_agents
    gru_h = [[None, None]] * number_of_agents
    states_for_forward_pass = [None] * number_of_agents
    for episode in range(1000):
        observations = env.reset()
        done = False
        # @TODO(Yaniv): there is abatch of games so you need to consider it when saving into replaybuffer and when you perform done condition!!
        while not done:
            actions = []

            for i in range(2):
                obs = observations[i]

                message = messages[(i + 1) % 2]
                u_tm1 = actions_tm1[i]  # this is the previouse action of the agent
                a = [i] * batch_size  # this is the agent id
                h_1 = gru_h[i][0]
                h_2 = gru_h[i][1]
                states_for_forward_pass[i] = (obs, message, u_tm1, a, h_1, h_2)

                action, message, out2, h1, h2 = agent_1.get_action_and_message(
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

                gru_h[i][0] = h1
                gru_h[i][1] = h2
                actions_tm1[i] = action
                messages[i] = message

            next_observations, reward, done, info = env.step(actions)

            state_1 = states_for_forward_pass[0]
            state_2 = states_for_forward_pass[1]

            next_states = [None] * number_of_agents
            for i in range(number_of_agents):

                next_obs = next_observations[i]
                next_message = messages[(i + 1) % 2]
                u_tm1 = actions_tm1[i]  # this is the previouse action of the agent
                a = i  # this is the agent id
                h_1 = gru_h[i][0]
                h_2 = gru_h[i][1]

                next_states[i] = (next_obs, next_message, u_tm1, a, h_1, h_2)

            observations = next_observations

            ## Here you should take into account the number of batches
            for b in range(batch_size):
                ## For  Agent 1
                agent_1.add_to_replay_buffer(
                    state=(
                        state_1[0][b].unsqueeze(0),  # obs
                        state_1[1][b],  # message
                        state_1[2][b],  # u_tm1
                        state_1[3],  # a (agent id, not batched)
                        state_1[4][b] if state_1[4] is not None else None,  # h_1
                        state_1[5][b] if state_1[5] is not None else None,  # h_2
                    ),
                    action=actions[b][0],
                    reward=reward[b][0],
                    next_state=(
                        next_states[0][0][b].unsqueeze(0),
                        next_states[0][1][b],
                        next_states[0][2][b],
                        next_states[0][3],
                        next_states[0][4][b] if next_states[0][4] is not None else None,
                        next_states[0][5][b] if next_states[0][5] is not None else None,
                    ),
                    done=done,
                )
                # For agent 2
                agent_2.add_to_replay_buffer(
                    state=(
                        state_2[0][b].unsqueeze(0),
                        state_2[1][b],
                        state_2[2][b],
                        state_2[3],
                        state_2[4][b] if state_2[4] is not None else None,
                        state_2[5][b] if state_2[5] is not None else None,
                    ),
                    action=actions[b][1],
                    reward=reward[b][1],
                    next_state=(
                        next_states[1][0][b].unsqueeze(0),
                        next_states[1][1][b],
                        next_states[1][2][b],
                        next_states[1][3],
                        next_states[1][4][b] if next_states[1][4] is not None else None,
                        next_states[1][5][b] if next_states[1][5] is not None else None,
                    ),
                    done=done[b] if hasattr(done, "__getitem__") else done,
                )


if __name__ == "__main__":
    main()
