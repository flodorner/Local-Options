import numpy as np
import random


def rand_argmax(x):
    return random.choice(np.argwhere(x == np.max(x)))[0]


def Qstar_to_values(Qs):
    out = {}
    for key in Qs:
        out[key] = np.max(Qs[key])
    return out


def Qs_to_policy(Qs, Q_trafo=lambda x: x):
    policy = {}
    for key in Qs:
        policy[key] = np.argmax(Q_trafo(Qs[key]))
    return policy


def one_hot(i, n):
    a = np.zeros(n)
    a[i] = 1
    return a


# Output: value map maxQs, i=> value of i

def Multi_Qs_to_F(policy, Qs, exits):
    # outputs: entries to S_{1)
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
    # outputs: entries to S_{1)
    coeffs = []
    for entry in exits:
        V = np.array(Vs[entry])
        coeffs.append(V)

    coeffs = np.array(coeffs)
    W = coeffs[:, 1:]
    B = coeffs[:, 0]
    return lambda v, i: coeffs[i][0] + sum([v[j] * coeffs[i][j + 1] for j in range(len(coeffs[i]) - 1)]), W, B





