import torch
import torch.nn as nn

class GuessMyNumberEnv:
    def __init__(self, batch_size=1, image_size=16, n_channels=3, max_steps = 4):
        self.bs = batch_size
        self.image_size = image_size
        self.n_channels = n_channels
        self.action_space = 5
        self.max_steps = max_steps
        self.reset()

    def _generate_dummy(self):
        digits = torch.randint(0, self.action_space, (self.bs,))
        digits = digits.reshape(self.bs, 1) #(batch, 1)
        return  digits

    def reset(self):
        self.step_count = 0
        
        self.state1 = self._generate_dummy()
        self.state2 = self._generate_dummy()
        return self.state1.float(), self.state2.float()

    def get_digits(self):
        return self.state1[1], self.state2[1]

    def get_reward(self, a1, a2):
        if self.step_count >= self.max_steps:
            d1 = self.state1
            d2 = self.state2
            r1 = (a1 == d2).float()
            r2 = (a2 == d1).float()
        else:
            r1  = torch.zeros(self.bs, 1)
            r2  = torch.zeros(self.bs, 1)

        return r1, r2

    def step(self, a1, a2):

        self.step_count += 1
        done = self.step_count >= self.max_steps
       
        r1, r2 = self.get_reward(a1, a2)
        r = [r1, r2]
        state = (self.state1.float(), self.state2.float())
        return state,r, done



if __name__ == "__main__":
    env = GuessMyNumberEnv()
    obs1, obs2 = env.reset()
    print("Initial Observations:", obs1, obs2)

    for step in range(env.max_steps):
        a1 = torch.randint(0, env.action_space, (env.bs,))
        a2 = torch.randint(0, env.action_space, (env.bs,))
        print(f"Step {step + 1}: Actions - Agent 1: {a1}, Agent 2: {a2}")

        state, rewards, done = env.step(a1, a2)
        print(f"Rewards - Agent 1: {rewards[0]}, Agent 2: {rewards[1]}")
        
        if done:
            print("Episode finished.")
            break