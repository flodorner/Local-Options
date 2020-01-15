import numpy as np
import random


class partial_env_A:
    def __init__(self, env, entries, exits, values=None, gamma=0.99, track_exits=False, randomstart=True):
        # Implements a local MDP with single on the part of the state space confined by entries and exits.
        # values specifies the additional rewards that are obtained by reaching each of the exit states.
        # If track_exits is True, the distribution of exits is saved as a parameter and can be used to initialize
        # the complementary local MDP on the rest of the state space. If randomstart is True, the start state is sampled
        # from the entries. This requires that the original MDP can be reset to this state.
        self.env = env

        self.entries = entries
        self.exits = exits

        self.gamma = gamma
        self.dynamic = False
        self.track_exits = track_exits
        if track_exits is True:
            self.exit_distribution = {}

        if values is None:
            self.values = [0 for i in range(len(self.exits))]
        else:
            self.values = values

    def reset(self):
        self.env.reset()
        if self.randomstart:
            self.env.state = random.choice(self.entries)
        if self.track_exits is True:
            self.last_entry = self.env.state
        return self.env.state

    def step(self, act, Qs=None):
        obs, rew, done, _ = self.env.step(act)
        if obs in self.exits:
            i = self.exits.index(obs)
            done = True
            rew = rew + self.gamma * self.values[i]
            if self.track_exits is True:
                try:
                    self.exit_distribution[self.last_entry][i] = self.exit_distribution[self.last_entry][i] + 1
                except:
                    self.exit_distribution[self.last_entry] = np.zeros(len(self.exits))
                    self.exit_distribution[self.last_entry][i] = self.exit_distribution[self.last_entry][i] + 1

        return obs, rew, done, None


class partial_env_A_multireward:
    def __init__(self, env, entries, exits, track_exits=False, randomstart=True, gamma=0.99):
        # Implements a local MDP with multiple rewards on the part of the state space confined by entries and exits.
        # The first reward is the original one and the other rewards are proxy rewards given for reaching the respective
        # exits. If track_exits is True, the distribution of exits is saved as a parameter and can be used to initialize
        # the complementary local MDP on the rest of the state space. If randomstart is True, the start state is sampled
        # from the entries. This requires that the original MDP can be reset to this state.
        self.env = env

        self.entries = entries
        self.exits = exits
        self.dynamic = False
        self.track_exits = track_exits
        self.randomstart = randomstart
        self.gamma = gamma
        if track_exits is True:
            self.exit_distribution = {}

    def reset(self):
        self.env.reset()
        if self.randomstart:
            self.env.state = random.choice(self.entries)
        if self.track_exits is True:
            self.last_entry = self.env.state
        return self.env.state

    def step(self, act, Qs=None):
        obs, rew, done, _ = self.env.step(act)
        rews = np.zeros(len(self.exits) + 1)
        rews[0] = rew
        if obs in self.exits:
            i = self.exits.index(obs)
            done = True
            rews[i + 1] = self.gamma
            if self.track_exits is True:
                try:
                    self.exit_distribution[self.last_entry][i] = self.exit_distribution[self.last_entry][i] + 1
                except:
                    self.exit_distribution[self.last_entry] = np.zeros(len(self.exits))
                    self.exit_distribution[self.last_entry][i] = self.exit_distribution[self.last_entry][i] + 1

        return obs, rews, done, None


class partial_env_B:
    def __init__(self, env, entries, exits, value_map, actions, gamma=0.99, entry_distribution=None):
        # Implements a local MDP on the part of the state space confined by entries and exits.
        # Value map gets the Qs for the entry states and an index i and returns the ith entry of
        # the operator F applied to the entry Qs and is used to give bonus rewards for transitions to the exits.
        # Theoretically, this might be doable by a neural net that maps states to affine functions and applies
        # them to the Qs, instead. entry_distribution can be used to define a different start distribution than in the
        # original MDP. This is necessary, if the original MDP starts outside of the treated part.

        self.env = env

        self.entries = entries
        self.exits = exits

        self.gamma = gamma
        self.dynamic = True
        self.value_map = value_map
        self.entry_distribution = entry_distribution
        self.actions = actions

    def reset(self):
        self.env.reset()
        if self.entry_distribution is not None:
            self.env.state = np.random.choice(self.entries, p=self.entry_distribution)
        return self.env.state

    def step(self, act, Qs):
        obs, rew, done, _ = self.env.step(act)
        if obs in self.exits:
            i = self.exits.index(obs)
            done = True
            for s in self.entries:
                try:
                    Qs[s]
                except:
                    Qs[s] = [0 for i in range(self.actions)]
            value = self.value_map([max(Qs[s]) for s in self.entries], i)
            rew = rew + self.gamma * value
        return obs, rew, done, None
