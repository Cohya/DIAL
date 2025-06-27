import numpy as np
import random
import os 
import sys 

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
    
from nn_models.learn2com.utils.rb_abstract import RBAbstract

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # sum tree nodes
        self.data = np.zeros(capacity, dtype=object)  # actual transitions
        self.size = 0
        self.pointer = 0

    def add(self, priority, data):
        index = self.pointer + self.capacity - 1
        self.data[self.pointer] = data
        self.update(index, priority)

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, index, priority):
        change = priority - self.tree[index]
        self.tree[index] = priority
        self._propagate(index, change)

    def _propagate(self, index, change):
        parent = (index - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get_leaf(self, value):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self):
        return self.tree[0]

class PrioritizedReplayBuffer(RBAbstract):
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        self.epsilon = 1e-5
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority ** self.alpha, transition)

    def sample(self, batch_size, frame_idx):
        batch = []
        idxs = []
        segment = self.tree.total / batch_size
        priorities = []

        self.beta = min(1.0, self.beta + self.beta_increment)
        # self.beta = min(1.0, self.beta_start + frame_idx * self.beta_increment)

        for i in range(batch_size):
            s = random.uniform(i * segment, (i + 1) * segment)
            idx, priority, data = self.tree.get_leaf(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        sampling_probs = np.array(priorities) / self.tree.total
        weights = (self.tree.size * sampling_probs) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(idxs),
            np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            priority = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.size



# if __name__ == "__main__":
#     replay_buffer = PrioritizedReplayBuffer(capacity=10000)

# def train_step(policy_net, target_net, optimizer, replay_buffer, frame_idx):
#     if len(replay_buffer) < BATCH_SIZE:
#         return

#     states, actions, rewards, next_states, dones, idxs, weights = replay_buffer.sample(BATCH_SIZE, frame_idx)

#     states = torch.FloatTensor(states)
#     actions = torch.LongTensor(actions)
#     rewards = torch.FloatTensor(rewards)
#     next_states = torch.FloatTensor(next_states)
#     dones = torch.FloatTensor(dones)
#     weights = torch.FloatTensor(weights)

#     q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
#     with torch.no_grad():
#         next_q_values = target_net(next_states).max(1)[0]
#         target_q = rewards + GAMMA * next_q_values * (1 - dones)

#     td_errors = q_values - target_q
#     loss = (td_errors.pow(2) * weights).mean()

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     replay_buffer.update_priorities(idxs, td_errors.detach().numpy())
