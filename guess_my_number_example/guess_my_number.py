import numpy as np 


class GuessMyNumberEnv:
    def __init__(self, max_steps = 4, action_space = 5):
        self.action_space = action_space
        self.max_steps = max_steps

    def _generate_dummy(self):
        self.player_1_digit = np.random.randint(0, self.action_space, (1,1))
        self.player_2_digit = np.random.randint(0, self.action_space, (1,1))

        return  self.player_1_digit, self.player_2_digit

    def reset(self):
        self.step_count = 0
        self.state1, self.state2 = self._generate_dummy()
        obs = {"agent_1": self.state1, "agent_2": self.state2}
        return obs

    def get_digits(self):
        return self.state1[1], self.state2[1]

    def get_reward(self, a1, a2):
        if self.step_count >= self.max_steps:
            d1 = self.state1
            d2 = self.state2
            r1 = float(a1 == d2[0][0])
            r2 = float(a2 == d1[0][0])
        else:
            r1 = 0.0
            r2 = 0.0
        return r1, r2

    def step(self, a1, a2):
        self.step_count += 1
        done = self.step_count >= self.max_steps
       
        r1, r2 = self.get_reward(a1, a2)
        r = [r1, r2]
        obs = {"agent_1": self.state1, "agent_2": self.state2}
        return obs,r, done



if __name__ == "__main__":
    env = GuessMyNumberEnv()
    obs1, obs2 = env.reset()
    print("Initial Observations:", obs1, obs2)

    obs1, obs2 = env.reset()
    done = False
    step_count = 0
    while not done:
        a1 = np.random.randint(0, env.action_space, 1)
        a2 = np.random.randint(0, env.action_space, 1)
        state, rewards, done = env.step(a1, a2)
        print(f"Rewards - Agent 1: {rewards[0]}, Agent 2: {rewards[1]}")
        step_count += 1


    print(f"Step count: {step_count}")