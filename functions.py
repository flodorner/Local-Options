import numpy as np
import random


def rand_argmax(x):
    # Returns an index that is randomly sampled from the indices with maximial entries in an array.
    return random.choice(np.argwhere(x == np.max(x)))[0]


def Qstar_to_values(Qs):
    #Takes a dict of Q* values and returns a dict of V* values.
    out = {}
    for key in Qs:
        out[key] = np.max(Qs[key])
    return out


def Qs_to_policy(Qs, Q_trafo=lambda x: x):
    # Takes a dict of Q values and returns the greedy policy as dict
    policy = {}
    for key in Qs:
        policy[key] = np.argmax(Q_trafo(Qs[key]))
    return policy


def one_hot(i, n):
    # Takes a number i and a length n and returns the one-hot encoding for the number of the length
    a = np.zeros(n)
    a[i] = 1
    return a

def Multi_Qs_to_F(policy, Qs, exits):
    # Returns the operator F, given a policy, Q-values and a list of exit states.
    coeffs = []
    for entry in exits:
        try:
            Q = np.array(Qs[entry])
        except:
            Q = [[0 for i in exits] for i in range(len(Qs[list(Qs.keys())[0]]))]
        try:
            V = np.average(Q, weights=policy[entry], axis=0)
        except:
            V = Q[policy[entry]]
        if np.all(V == 0):
            V = [0 for i in exits]
        coeffs.append(V)

    coeffs = np.array(coeffs)
    W = coeffs[:, 1:]
    B = coeffs[:, 0]
    return lambda v, i: coeffs[i][0] + sum([v[j] * coeffs[i][j + 1] for j in range(len(coeffs[i]) - 1)]), W, B


def Multi_Vs_to_F(policy, Vs, exits):
    # Returns the operator F, given a policy, V-values and a list of exit states.
    coeffs = []
    for entry in exits:
        V = np.array(Vs[entry])
        coeffs.append(V)

    coeffs = np.array(coeffs)
    W = coeffs[:, 1:]
    B = coeffs[:, 0]
    return lambda v, i: coeffs[i][0] + sum([v[j] * coeffs[i][j + 1] for j in range(len(coeffs[i]) - 1)]), W, B





