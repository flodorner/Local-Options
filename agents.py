import numpy as np


class policy_agent_deterministic:
    def __init__(self, policy, num_act):
        # Implements a simple agent that carries out a determinstic policy.
        # policy is a dict that maps states to actions. If the policy is not defined, the agent acts randomly.
        self.policy = policy
        self.num_act = num_act

    def act(self, obs):
        try:
            return self.policy[obs]
        except:
            return np.random.choice(np.arange(self.num_act))


class policy_agent:
    # Implements a simple agent that carries out a policy.
    # policy is a dict that maps states to distributions over actions (in form of a list of probabilities).
    def __init__(self, policy, num_act):
        self.policy = policy
        self.num_act = num_act

    def act(self, obs):
        return np.random.choice(np.arange(self.num_act), p=self.policy[obs])
