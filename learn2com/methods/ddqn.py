import os
import sys
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from nn_models.learn2com.utils.discretise_regularise_unit import (
    get_discretise_regularise_unit,
)
from nn_models.learn2com.utils.rb_abstract import RBAbstract


class RL_model(ABC):

    @abstractmethod
    def get_action_and_message(self, state):
        """Return an action based on the current state."""
        pass

    @abstractmethod
    def train_step(self, frame_idx: int, batch_size: int):
        """Perform a training step."""
        pass

    @abstractmethod
    def save(self, path):
        """Save the model to the specified path."""
        pass

    @abstractmethod
    def load(self, path):
        """Load the model from the specified path."""
        pass


class DDQN(RL_model):
    def __init__(
        self,
        main_net: torch.nn.Module,
        target_net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float,
        i_d: int,
    ):

        self.main_net = main_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.gamma = gamma
        self.action_dims = self.main_net.action_dims
        self.obs_dims = self.main_net.obs_dims
        self.did_backward = False

    def get_action_and_message(self, **kwargs):
        output = self.main_net(**kwargs)
        return output

    def train_step(self, frame_idx: int, batch_size: int, replay_buffer: RBAbstract):
        if len(replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones, idxs, weights = (
            replay_buffer.sample(batch_size, frame_idx)
        )

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        weights = torch.FloatTensor(weights)

        # DDQN: use main_net to select action, target_net to evaluate
        with torch.no_grad():
            next_actions = self.main_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = (
                self.target_net(next_states).gather(1, next_actions).squeeze()
            )
            target_q = rewards + self.gamma * next_q_values * (1 - dones)

        q_values = self.main_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        td_errors = q_values - target_q
        loss = (td_errors.pow(2) * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        replay_buffer.update_priorities(idxs, td_errors.detach().numpy())

    def save(self, path):
        torch.save(self.main_net.state_dict(), path)

    def load(self, path):
        self.main_net.load_state_dict(torch.load(path))

    def update_target_net(self):
        self.target_net.load_state_dict(self.main_net.state_dict())

    def prepare_data_for_network(self, agent_rb: Tuple):
        r_vec = []
        done_vec = []
        next_obs_vec = []
        next_message_vec = []
        next_u_tm1_vec = []
        next_a_vec = []
        next_h_1_vec = []
        next_h_2_vec = []

        obs_vec = []
        message_vec = []
        u_tm1_vec = []
        a_vec = []
        h_1_vec = []
        h_2_vec = []
        action_vec = []
        for j in range(len(agent_rb)):
            single_mem = agent_rb[j]
            # (state, action, reward, next_state, done)
            # state =  (obs, message', u_tm1, a, h_1, h_2)

            r = single_mem[2]
            r_vec.append(r)
            a_t = single_mem[1]
            action_vec.append(a_t)
            done = single_mem[4]
            done_vec.append(done)

            next_obs = single_mem[3][0]  # obs (B, 3, 28,28)
            next_message = single_mem[3][1]  # (B, 1) # message'
            next_u_tm1 = single_mem[3][2]  #  (B, 1) - previouse avtions
            next_a = single_mem[3][3]  # (B, 1) agent id
            next_h_1 = single_mem[3][4]  # h1 (1, B , embedding dim)
            next_h_2 = single_mem[3][5]  # h2 (1, B , embedding dim)

            next_obs_vec.append(next_obs)
            next_message_vec.append(next_message.unsqueeze(0))
            next_u_tm1_vec.append(next_u_tm1)
            next_a_vec.append(next_a)
            next_h_1_vec.append(next_h_1)
            next_h_2_vec.append(next_h_2)

            obs = single_mem[0][0]  # obs (B, 3, 28,28)
            message = single_mem[0][1]  # (B, 1) # message'
            u_tm1 = single_mem[0][2]  #  (B, 1) - previouse avtions
            a = single_mem[0][3]  # (B, 1) agent id
            h_1 = single_mem[0][4]  # h1 (1, B , embedding dim)
            h_2 = single_mem[0][5]  # h2 (1, B , embedding dim)

            obs_vec.append(obs)
            message_vec.append(message.unsqueeze(0))
            u_tm1_vec.append(u_tm1)
            a_vec.append(a)
            h_1_vec.append(h_1)
            h_2_vec.append(h_2)

        # conv ert to tgensors
        ## Next Obs
        next_obs = torch.cat(next_obs_vec, dim=0)
        next_message = torch.cat(next_message_vec, dim=0)
        next_u_tm1 = torch.stack(next_u_tm1_vec)
        next_a = torch.Tensor(next_a_vec)
        next_h_1 = torch.stack(next_h_1_vec, dim=1)
        next_h_2 = torch.stack(next_h_2_vec, dim=1)

        ######
        obs = torch.cat(obs_vec, dim=0)
        message = torch.cat(message_vec, dim=0)
        u_tm1 = torch.stack(u_tm1_vec)
        a = torch.Tensor(a_vec)
        h_1 = torch.stack(h_1_vec, dim=1)
        h_2 = torch.stack(h_2_vec, dim=1)

        return (
            (obs, message, u_tm1, a, h_1, h_2),
            (next_obs, next_message, next_u_tm1, next_a, next_h_1, next_h_2),
            r_vec,
            done_vec,
            action_vec,
        )

    def get_grad_theta_wrt_q(self, agent_rb: Tuple) -> List[torch.Tensor]:
        self.did_backward = False
        (
            (obs, message, u_tm1, a, h_1, h_2),
            (next_obs, next_message, next_u_tm1, next_a, next_h_1, next_h_2),
            r_vec,
            done_vec,
            action_vec,
        ) = self.prepare_data_for_network(agent_rb)

        q_target = self.get_q_target(
            next_obs,
            next_message,
            next_u_tm1,
            next_a,
            next_h_1,
            next_h_2,
            done_vec,
            r_vec,
        )

        q_prediction = self.get_q_estimated(
            obs, message, u_tm1, a, h_1, h_2, action_vec
        )

        loss = self.calculate_loss(q_prediction, q_target)
        # Backward: calculate the gradients
        loss.backward()
        self.did_backward = True
        delta_theta = []
        for param in self.main_net.parameters():
            if param.grad is not None:
                delta_theta.append(param.grad.clone())
            else:
                delta_theta.append(None)

        return delta_theta

    def calculate_loss(self, q_prediction, q_target):
        loss = F.mse_loss(q_prediction, q_target)
        return loss

    def get_q_estimated(
        self, obs, message, u_tm1, a, h_1, h_2, action_vec, message_require_grad=False
    ):

        if not message_require_grad:
            message = message.detach()
        q_values_main, _, _, _, _ = self.main_net(
            obs=obs,  # obs
            message=message,  # message' <-- message.requires_grad = True, detach() cancle  the gradient
            u_tm1=u_tm1,  # u
            a=a,  # a
            h_1=h_1.detach(),  # h1 <- .requires_grad = True, detach cancle the gradients
            h_2=h_2.detach(),  # h2 <- .requires_grad = True, detach cancle the gradients
        )  # (B, num_actions)

        action_vec = torch.stack(action_vec).long()
        num_classes = self.action_dims
        one_hot = F.one_hot(action_vec, num_classes=num_classes)  # [B, num_classes]

        q_prediction = q_values_main * one_hot
        q_prediction = torch.sum(q_prediction, dim=1)

        return q_prediction

    def get_q_target(
        self,
        next_obs,
        next_message,
        next_u_tm1,
        next_a,
        next_h_1,
        next_h_2,
        done_vec,
        r_vec,
    ):

        q_values_target, _, _, _, _ = self.target_net(
            obs=next_obs,  # obs
            message=next_message,  # message'
            u_tm1=next_u_tm1,  # u
            a=next_a,  # a
            h_1=next_h_1,  # h1
            h_2=next_h_2,  # h2
        )
        q_max = q_values_target.max(1, keepdim=True)[0].detach()
        done = torch.stack(done_vec)
        r = torch.stack(r_vec)
        q_targets = r + self.gamma * q_max.squeeze(1) * (1 - done)

        return q_targets

    def get_grad_theta_wrt_message(self, agent_rb: List[Tuple]):

        (
            (obs, message, u_tm1, a, h_1, h_2),
            (next_obs, next_message, next_u_tm1, next_a, next_h_1, next_h_2),
            r_vec,
            done_vec,
            action_vec,
        ) = self.prepare_data_for_network(agent_rb)

        q_target = self.get_q_target(
            next_obs,
            next_message,
            next_u_tm1,
            next_a,
            next_h_1,
            next_h_2,
            done_vec,
            r_vec,
        )

        message = message.requires_grad_()
        message.retain_grad()  # ensures .grad is populated even if not leaf
        q_prediction = self.get_q_estimated(
            obs, message, u_tm1, a, h_1, h_2, action_vec, message_require_grad=True
        )

        loss = self.calculate_loss(q_prediction, q_target)
        # Backward: calculate the gradients
        loss.backward()
        self.did_backward = True
        # find teh derivative of loss with respect to message
        message_gradient = message.grad.detach()

        # # free the graph to save memory
        # message.requires_grad = False
        return message_gradient

    def get_grad_message_wrt_message(self, agent_rb: List[Tuple]):
        (
            (obs, message, u_tm1, a, h_1, h_2),
            (next_obs, next_message, next_u_tm1, next_a, next_h_1, next_h_2),
            r_vec,
            done_vec,
            action_vec,
        ) = self.prepare_data_for_network(agent_rb)

        message = message.detach().requires_grad_()

        _, message_of_the_agent, _, _, _ = self.main_net(
            obs, message, u_tm1, a, h_1, h_2
        )

        # find teh derivative of loss with respect to message
        # Suppose you want the sum of message_of_the_agent (or any scalar function of it)
        massgae_2_message_grad_all_outputs = torch.autograd.grad(
            outputs=message_of_the_agent,
            inputs=message,
            grad_outputs=torch.ones_like(message_of_the_agent),
            retain_graph=False,
            create_graph=True,
            allow_unused=True,
        )
        massgae_2_message_grad = massgae_2_message_grad_all_outputs[0]

        return massgae_2_message_grad.detach().clone(), message.detach().clone()

    def get_grad_of_dru_mat_wrt_theta(
        self, agent_rb: List[Tuple], DRU: Callable[[torch.Tensor, bool], torch.Tensor]
    ):
        # Callable[[torch.Tensor, bool], torch.Tensor] is hint to say that this is a function
        # which its inputs are torch.Tensor and bool and its output is torch.Tensor
        (
            (obs, message, u_tm1, a, h_1, h_2),
            (next_obs, next_message, next_u_tm1, next_a, next_h_1, next_h_2),
            r_vec,
            done_vec,
            action_vec,
        ) = self.prepare_data_for_network(agent_rb)

        _, message_output, _, _, _ = self.main_net(
            obs=obs,  # obs
            message=message,  # message' <-- message.requires_grad = True, detach() cancle  the gradient
            u_tm1=u_tm1,  # u
            a=a,  # a
            h_1=h_1.detach(),  # h1 <- .requires_grad = True, detach cancle the gradients
            h_2=h_2.detach(),  # h2 <- .requires_grad = True, detach cancle the gradients
        )  # (B, num_actions)

        dru_message = DRU(message_output, training=True)

        # Zero gradients before backward (this is what optimizer.zero_grad do inside the optimizer)
        self.main_net.zero_grad()

        # Backward from dru_message (make sure it's a scalar or sum it)
        dru_message.sum().backward(retain_graph=False)

        ### We can do like: if we want the gradients of the DRU w.r.t. theta for each sample  in the batch
        # batch_size = dru_message.shape[0]
        # per_sample_grads = []

        # for i in range(batch_size):
        #     self.main_net.zero_grad()
        #     dru_message[i].backward(retain_graph=True)
        #     grads = []
        #     for name, param in self.main_net.named_parameters():
        #         if param.grad is not None:
        #             grads.append(param.grad.clone())
        #         else:
        #             grads.append(None)
        #     per_sample_grads.append(grads)
        ## Now we have the gradients of the DRU wrt theta
        grads_of_dru_wrt_thets = []
        for (
            name,
            param,
        ) in self.main_net.named_parameters():  # self.main_net.parameters():
            # print(name)
            if param.grad is not None:
                grads_of_dru_wrt_thets.append(param.grad.clone())
            else:
                grads_of_dru_wrt_thets.append(None)

                # Collect gradients w.r.t. model parameters
        return grads_of_dru_wrt_thets

    def copy_weights_from_other_network(self, other_network: nn.Module):

        for param, other_param in zip(
            self.main_net.parameters(), other_network.parameters()
        ):
            param.data.copy_(other_param.data)
