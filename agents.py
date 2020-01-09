import numpy as np
import random


class policy_agent_deterministic:
    def __init__(self, policy, num_act):
        self.policy = policy
        self.num_act = num_act

    def act(self, obs):
        try:
            return self.policy[obs]
        except:
            return np.random.choice(np.arange(self.num_act))


class policy_agent:
    def __init__(self, policy, num_act):
        self.policy = policy
        self.num_act = num_act

    def act(self, obs):
        return np.random.choice(np.arange(self.num_act), p=self.policy[obs])
