import random
import numpy as np


def generate_rewards(n, actions):
    return np.random.uniform(size=(n, actions, n))


def generate_transitions(n, actions):
    M = np.zeros((actions, n, n))
    for act in range(actions):
        for start in range(n):
            p = np.random.uniform(size=(n))
            p = p / sum(p)
            M[act][start] = p
    return M


class MDP:
    def __init__(self, M, R, ps=0):
        self.episodes = 0
        self.M = M
        self.R = R
        self.ps = ps

    def reset(self):
        if isinstance(self.ps, int):
            self.state = self.ps
        else:
            self.state = np.random.choice(np.arange(len(self.ps)), p=self.ps)
        self.episodes += 1
        return self.state

    def step(self, act):
        new_state = np.random.choice(np.arange(len(self.M[act])), p=self.M[act][self.state])
        reward = self.R[new_state][act][self.state]
        self.state = new_state

        done = False

        # detect terminal?!

        return self.state, reward, done, None

    def uniform_policy_operator(self, start_B, gamma=0.99):

        P = np.mean(self.M, axis=0)
        R = np.mean(self.R, axis=1)
        R = np.diagonal(np.matmul(P, R))
        R = R[:start_B]

        E = P[:start_B, start_B:]
        P = P[:start_B, :start_B]

        """
        print( np.linalg.norm( np.linalg.inv( np.eye(P.shape[0])-0.99*P ),ord=np.inf )  )
        print(np.linalg.norm(np.eye(P.shape[0])-
                              np.matmul(
                                  np.linalg.inv( np.eye(P.shape[0])-0.99*P ),
                                                            ( np.eye(P.shape[0])-0.99*P )),
                             ord=np.inf)*np.linalg.norm(np.eye(P.shape[0])-0.99*P,ord=np.inf))
        """

        B_True = np.matmul(np.linalg.inv((np.eye(P.shape[0]) - gamma * P)), R)
        W_True = np.matmul(np.linalg.inv((np.eye(P.shape[0]) - gamma * P)), gamma * E)
        return W_True, B_True


def random_MDP(n, acts, B_start):
    M = generate_transitions(n, acts)
    R = generate_rewards(n, acts)
    env = MDP(M, R)
    W, B = env.uniform_policy_operator(B_start)
    return env, W, B


class robot():
    def __init__(self):
        self.state = (16, 0, 0)
        self.episodes = 0
        self.steps = 0

    def reset(self):
        self.state = (16, 0, 0)
        self.episodes += 1
        return self.state

    def step(self, act):
        self.steps += 1
        rew = 0
        if act == 0:
            if self.state[0] > 1:
                self.state = (self.state[0] - 1, self.state[1], self.state[2])
                rew += 1
            elif self.state[0] > 0:
                None
            else:
                # stop charging
                self.state = (-self.state[0], self.state[1], self.state[2])
        if act == 1:
            if self.state[0] > 4:
                self.state = (self.state[0] - 5, self.state[1] + 1, self.state[2])
                if self.state[1] > 1:
                    rew += 10
                    self.state = (self.state[0], self.state[1], self.state[2])
            elif self.state[0] > 0:
                None
            else:
                # stop charging
                self.state = (-self.state[0], self.state[1], self.state[2])
        if act == 2:
            if self.state[0] > 4:
                self.state = (self.state[0] - 5, self.state[1], self.state[2] + 1)
                if self.state[2] == 3:
                    rew += 100
            elif self.state[0] > 0:
                None
            else:
                # stop charging
                self.state = (-self.state[0], self.state[1], self.state[2])
        # Implement this consistently!!!
        if act == 3:
            if self.state[0] > 0 or (self.state[0]==0 and sum([self.state[1],self.state[2]])>0):
                self.state = (-self.state[0], 0, 0)
            elif self.state[0] < -11:
                self.state = (self.state[0] - 3, 0, 0)
                self.state = (max((self.state[0], -16)), 0, 0)
            elif self.state[0] < -1:
                self.state = (self.state[0] - 4, 0, 0)
            else:
                self.state = (self.state[0] - 9, 0, 0)
        return self.state, rew, False, None

