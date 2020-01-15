import numpy as np
from matplotlib import pyplot as plt

from agents import policy_agent
from algorithms import Tabular_q, SARSA, get_F, learn_matrices, Tdzero
from environments import random_MDP
from functions import Multi_Qs_to_F, Multi_Vs_to_F
from misc import compareplot
from wrapper import partial_env_A_multireward

np.random.seed(0)

Ws=[]
Bs=[]
for i in range(10000):
    env, W_True, B_True = random_MDP(50, 4, 46)
    Ws.append(np.linalg.norm(W_True,ord=np.inf))
    Bs.append(np.linalg.norm(B_True,ord=np.inf))
B50=np.mean(Bs)
W50=np.mean(Ws)

Ws=[]
Bs=[]
for i in range(10000):
    env, W_True, B_True = random_MDP(10, 4, 6)
    Ws.append(np.linalg.norm(W_True,ord=np.inf))
    Bs.append(np.linalg.norm(B_True,ord=np.inf))
B10=np.mean(Bs)
W10=np.mean(Ws)





random_dict = {key: [0.25, 0.25, 0.25, 0.25] for key in range(100)}
random = policy_agent(random_dict, 4)

k = 25
cs = [0.5, 1, 10, 100, 1000, 10000]

QW = []
VW = []
QB = []
VB = []

for c in cs:
    sWs = []
    sBs = []
    for i in range(k):
        env, W_True, B_True = random_MDP(10, 4, 6)
        b = partial_env_A_multireward(env, np.arange(6), [6, 7, 8, 9])
        F, W, B = get_F(env, 1000, 4, random_dict
                        , np.arange(6), [6, 7, 8, 9], c=c, episode_length=100, epsilon=0,
                        alpha=lambda x, y: 0.1, evaluation="SARSA", re_evaluation_factor=1)
        sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
        sBs.append(np.linalg.norm(B - B_True, ord=np.inf))
    QW.append(sWs)
    QB.append(sBs)

    sWs = []
    sBs = []
    for i in range(k):
        env, W_True, B_True = random_MDP(10, 4, 6)
        b = partial_env_A_multireward(env, np.arange(6), [6, 7, 8, 9])
        F, W, B = get_F(env, 1000, 4, random_dict
                        , np.arange(6), [6, 7, 8, 9], c=c, episode_length=100, epsilon=0,
                        alpha=lambda x, y: 0.1, evaluation="td", re_evaluation_factor=1)
        sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
        sBs.append(np.linalg.norm(B - B_True, ord=np.inf))
    VW.append(sWs)
    VB.append(sBs)

sWs = []
sBs = []
for i in range(k):
    env, W_True, B_True = random_MDP(10, 4, 6)
    b = partial_env_A_multireward(env, np.arange(6), [6, 7, 8, 9])
    Qs = SARSA(b, 1000, 4, random, episode_length=100, epsilon=0, alpha=lambda x, y: 0.1)
    F, W, B = Multi_Qs_to_F(random_dict, Qs, np.arange(6))
    sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
    sBs.append(np.linalg.norm(B - B_True, ord=np.inf))

QdW = sWs
QdB = sBs

sWs = []
sBs = []
for i in range(k):
    env, W_True, B_True = random_MDP(10, 4, 6)
    b = partial_env_A_multireward(env, np.arange(6), [6, 7, 8, 9])
    Vs = Tdzero(b, 1000, 4, random, episode_length=100, epsilon=0, alpha=lambda x, y: 0.1)
    F, W, B = Multi_Vs_to_F(random_dict, Vs, np.arange(6))
    sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
    sBs.append(np.linalg.norm(B - B_True, ord=np.inf))

VdW = sWs
VdB = sBs

compareplot(np.log10(cs), [VW, QW, [VdW for i in cs], [QdW for i in cs],[[W10] for i in cs]],
            ["V_indirect", "Q_indirect", "V_direct", "Q_direct","True W"],
            ["b", "orange", "lightblue", "navajowhite", "red"],
            title="Random_MDP(10,4,6) 1000 episodes", xlabel="log 10 c", ylabel="error in W", ylim=None)

compareplot(np.log10(cs), [VB, QB, [VdB for i in cs], [QdB for i in cs],[[B10] for i in cs]],
            ["V_indirect", "Q_indirect", "V_direct", "Q_direct","True B"],
            ["b", "orange", "lightblue", "navajowhite", "red"],
            title="Random_MDP(10,4,6) 1000 episodes", xlabel="log 10 c", ylabel="error in B", ylim=None)


alpha = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
probW = []
QW = []
VW = []
probB = []
QB = []
VB = []
for a in alpha:
    sWs = []
    sBs = []
    for i in range(k):
        env, W_True, B_True = random_MDP(10, 8, 6)
        b = partial_env_A_multireward(env, np.arange(6), [6, 7, 8, 9])
        Vs = Tdzero(b, 1000, 4, random, episode_length=100, epsilon=0, alpha=lambda x, y: a)
        F, W, B = Multi_Vs_to_F(random_dict, Vs, np.arange(6))
        sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
        sBs.append(np.linalg.norm(B - B_True, ord=np.inf))
    VW.append(sWs)
    VB.append(sBs)

    sWs = []
    sBs = []
    for i in range(k):
        env, W_True, B_True = random_MDP(10, 8, 6)
        b = partial_env_A_multireward(env, np.arange(6), [6, 7, 8, 9])
        Qs = SARSA(b, 1000, 4, random, episode_length=100, epsilon=0, alpha=lambda x, y: a)
        F, W, B = Multi_Qs_to_F(random_dict, Qs, np.arange(6))
        sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
        sBs.append(np.linalg.norm(B - B_True, ord=np.inf))
    QW.append(sWs)
    QB.append(sBs)

compareplot(np.log10(alpha), [VW, QW ,[[W10] for i in alpha]],
            ["V", "Q","True W"], ["b", "orange", "red"],
            title="Random_MDP(10,4,6) 1000 episodes", xlabel="log 10 alpha", ylabel="error in W", ylim=None)

compareplot(np.log10(alpha), [VB, QB,[[B10] for i in alpha]],
            ["V", "Q","True B"], ["b", "orange", "red"],
            title="Random_MDP(10,4,6) 1000 episodes", xlabel="log 10 alpha", ylabel="error in B", ylim=None)

episodes = (np.arange(5)) * 500 + 100
probW = []
QW = []
VW = []
probB = []
QB = []
VB = []
for e in episodes:
    sWs = []
    sBs = []
    for i in range(k):
        env, W_True, B_True = random_MDP(10, 4, 6)
        b = partial_env_A_multireward(env, np.arange(6), [6, 7, 8, 9])
        F, W, B = learn_matrices(env, e, random, np.arange(6), [6, 7, 8, 9])
        sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
        sBs.append(np.linalg.norm(B - B_True, ord=np.inf))
    probW.append(sWs)
    probB.append(sBs)

    sWs = []
    sBs = []
    for i in range(k):
        env, W_True, B_True = random_MDP(10, 4, 6)
        b = partial_env_A_multireward(env, np.arange(6), [6, 7, 8, 9])
        Qs = SARSA(b, e, 4, random, episode_length=100, epsilon=0, alpha=lambda x, y: 0.1)
        F, W, B = Multi_Qs_to_F(random_dict, Qs, np.arange(6))
        sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
        sBs.append(np.linalg.norm(B - B_True, ord=np.inf))
    QW.append(sWs)
    QB.append(sBs)

    sWs = []
    sBs = []
    for i in range(k):
        env, W_True, B_True = random_MDP(10, 4, 6)
        b = partial_env_A_multireward(env, np.arange(6), [6, 7, 8, 9])
        Vs = Tdzero(b, e, 4, random, episode_length=100, epsilon=0, alpha=lambda x, y: 0.1)
        F, W, B = Multi_Vs_to_F(random_dict, Vs, np.arange(6))
        sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
        sBs.append(np.linalg.norm(B - B_True, ord=np.inf))
    VW.append(sWs)
    VB.append(sBs)

compareplot(episodes, [VW, QW, probW, [[W10] for i in episodes]],
            ["V", "Q", "Prob","True W"], ["b", "orange", "k","red"],
            title="Random_MDP(10,4,6) alpha=0.1", xlabel="episodes", ylabel="error in W", ylim=None)

compareplot(episodes, [VB, QB, probB,[[B10] for i in episodes]],
            ["V", "Q", "Prob","True B"], ["b", "orange", "k","red"],
            title="Random_MDP(10,4,6) alpha=0.1", xlabel="episodes", ylabel="error in B", ylim=None)


def alpha(v, t):
    return 1 / (v + 1)


episodes = (np.arange(5)) * 500 + 100
probW = []
QW = []
VW = []
probB = []
QB = []
VB = []
for e in episodes:
    sWs = []
    sBs = []
    for i in range(k):
        env, W_True, B_True = random_MDP(10, 4, 6)
        b = partial_env_A_multireward(env, np.arange(6), [6, 7, 8, 9])
        F, W, B = learn_matrices(env, e, random, np.arange(6), [6, 7, 8, 9])
        sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
        sBs.append(np.linalg.norm(B - B_True, ord=np.inf))
    probW.append(sWs)
    probB.append(sBs)

    sWs = []
    sBs = []
    for i in range(k):
        env, W_True, B_True = random_MDP(10, 4, 6)
        b = partial_env_A_multireward(env, np.arange(6), [6, 7, 8, 9])
        Qs = SARSA(b, e, 4, random, episode_length=100, epsilon=0, alpha=alpha)
        F, W, B = Multi_Qs_to_F(random_dict, Qs, np.arange(6))
        sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
        sBs.append(np.linalg.norm(B - B_True, ord=np.inf))
    QW.append(sWs)
    QB.append(sBs)

    sWs = []
    sBs = []
    for i in range(k):
        env, W_True, B_True = random_MDP(10, 4, 6)
        b = partial_env_A_multireward(env, np.arange(6), [6, 7, 8, 9])
        Vs = Tdzero(b, e, 4, random, episode_length=100, epsilon=0, alpha=alpha)
        F, W, B = Multi_Vs_to_F(random_dict, Vs, np.arange(6))
        sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
        sBs.append(np.linalg.norm(B - B_True, ord=np.inf))
    VW.append(sWs)
    VB.append(sBs)

compareplot(episodes, [VW, QW, probW, [[W10] for i in episodes]],
            ["V", "Q", "Prob","True W"], ["b", "orange", "k","red"],
            title="Random_MDP(10,4,6) alpha decay", xlabel="episodes", ylabel="error in W", ylim=None)

compareplot(episodes, [VB, QB, probB, [[B10] for i in episodes]],
            ["V", "Q", "Prob","True B"], ["b", "orange", "k","red"],
            title="Random_MDP(10,4,6) alpha decay", xlabel="episodes", ylabel="error in B", ylim=None)


def alpha(v, t):
    return 5 / (v + 5)


episodes = (np.arange(5)) * 500 + 100
probW = []
QW = []
VW = []
probB = []
QB = []
VB = []
for e in episodes:
    sWs = []
    sBs = []
    for i in range(k):
        env, W_True, B_True = random_MDP(10, 4, 6)
        b = partial_env_A_multireward(env, np.arange(6), [6, 7, 8, 9])
        F, W, B = learn_matrices(env, e, random, np.arange(6), [6, 7, 8, 9])
        sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
        sBs.append(np.linalg.norm(B - B_True, ord=np.inf))
    probW.append(sWs)
    probB.append(sBs)

    sWs = []
    sBs = []
    for i in range(k):
        env, W_True, B_True = random_MDP(10, 4, 6)
        b = partial_env_A_multireward(env, np.arange(6), [6, 7, 8, 9])
        Qs = SARSA(b, e, 4, random, episode_length=100, epsilon=0, alpha=alpha)
        F, W, B = Multi_Qs_to_F(random_dict, Qs, np.arange(6))
        sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
        sBs.append(np.linalg.norm(B - B_True, ord=np.inf))
    QW.append(sWs)
    QB.append(sBs)

    sWs = []
    sBs = []
    for i in range(k):
        env, W_True, B_True = random_MDP(10, 4, 6)
        b = partial_env_A_multireward(env, np.arange(6), [6, 7, 8, 9])
        Vs = Tdzero(b, e, 4, random, episode_length=100, epsilon=0, alpha=alpha)
        F, W, B = Multi_Vs_to_F(random_dict, Vs, np.arange(6))
        sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
        sBs.append(np.linalg.norm(B - B_True, ord=np.inf))
    VW.append(sWs)
    VB.append(sBs)

compareplot(episodes, [VW, QW, probW, [[W10] for i in episodes]],
            ["V", "Q", "Prob","True W"], ["b", "orange", "k","red"],
            title="Random_MDP(10,4,6) alpha decay 5", xlabel="episodes", ylabel="error in W", ylim=None)

compareplot(episodes, [VB, QB, probB, [[B10] for i in episodes]],
            ["V", "Q", "Prob","True B"], ["b", "orange", "k","red"],
            title="Random_MDP(10,4,6) alpha decay 5", xlabel="episodes", ylabel="error in B", ylim=None)


def alpha(v, t):
    return 1 / (v + 1)


episodes = (np.arange(5) + 1) * 2000
probW = []
QW = []
VW = []
probB = []
QB = []
VB = []
for e in episodes:
    sWs = []
    sBs = []
    for i in range(k):
        env, W_True, B_True = random_MDP(50, 4, 46)
        b = partial_env_A_multireward(env, np.arange(46), [46, 47, 48, 49])
        F, W, B = learn_matrices(env, e, random, np.arange(46), [46, 47, 48, 49])
        sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
        sBs.append(np.linalg.norm(B - B_True, ord=np.inf))
    probW.append(sWs)
    probB.append(sBs)

    sWs = []
    sBs = []
    for i in range(k):
        env, W_True, B_True = random_MDP(50, 4, 46)
        b = partial_env_A_multireward(env, np.arange(46), [46, 47, 48, 49])
        Qs = SARSA(b, e, 4, random, episode_length=100, epsilon=0, alpha=alpha)
        F, W, B = Multi_Qs_to_F(random_dict, Qs, np.arange(46))
        sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
        sBs.append(np.linalg.norm(B - B_True, ord=np.inf))
    QW.append(sWs)
    QB.append(sBs)

    sWs = []
    sBs = []
    for i in range(k):
        env, W_True, B_True = random_MDP(50, 4, 46)
        b = partial_env_A_multireward(env, np.arange(46), [46, 47, 48, 49])
        Vs = Tdzero(b, e, 4, random, episode_length=100, epsilon=0, alpha=alpha)
        F, W, B = Multi_Vs_to_F(random_dict, Vs, np.arange(46))
        sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
        sBs.append(np.linalg.norm(B - B_True, ord=np.inf))
    VW.append(sWs)
    VB.append(sBs)

compareplot(episodes, [VW, QW, probW, [[W50] for i in episodes]],
            ["V", "Q", "Prob","True W"], ["b", "orange", "k","red"],
            title="Random_MDP(50,4,46) alpha decay", xlabel="episodes", ylabel="error in W", ylim=None)

compareplot(episodes, [VB, QB, probB, [[B50] for i in episodes]],
            ["V", "Q", "Prob","True B"], ["b", "orange", "k","red"],
            title="Random_MDP(50,4,46) alpha decay", xlabel="episodes", ylabel="error in B", ylim=None)


def alpha(v, t):
    return 10 / (v + 10)


episodes = (np.arange(5) + 1) * 2000
probW = []
QW = []
VW = []
probB = []
QB = []
VB = []
for e in episodes:
    sWs = []
    sBs = []
    for i in range(k):
        env, W_True, B_True = random_MDP(50, 4, 46)
        b = partial_env_A_multireward(env, np.arange(46), [46, 47, 48, 49])
        F, W, B = learn_matrices(env, e, random, np.arange(46), [46, 47, 48, 49])
        sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
        sBs.append(np.linalg.norm(B - B_True, ord=np.inf))
    probW.append(sWs)
    probB.append(sBs)

    sWs = []
    sBs = []
    for i in range(k):
        env, W_True, B_True = random_MDP(50, 4, 46)
        b = partial_env_A_multireward(env, np.arange(46), [46, 47, 48, 49])
        Qs = SARSA(b, e, 4, random, episode_length=100, epsilon=0, alpha=alpha)
        F, W, B = Multi_Qs_to_F(random_dict, Qs, np.arange(46))
        sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
        sBs.append(np.linalg.norm(B - B_True, ord=np.inf))
    QW.append(sWs)
    QB.append(sBs)

    sWs = []
    sBs = []
    for i in range(k):
        env, W_True, B_True = random_MDP(50, 4, 46)
        b = partial_env_A_multireward(env, np.arange(46), [46, 47, 48, 49])
        Vs = Tdzero(b, e, 4, random, episode_length=100, epsilon=0, alpha=alpha)
        F, W, B = Multi_Vs_to_F(random_dict, Vs, np.arange(46))
        sWs.append(np.linalg.norm(W - W_True, ord=np.inf))
        sBs.append(np.linalg.norm(B - B_True, ord=np.inf))
    VW.append(sWs)
    VB.append(sBs)

compareplot(episodes, [VW, QW, probW ],
            ["V", "Q", "Prob"], ["b", "orange", "k"],
            title="Random_MDP(50,4,46) alpha decay 10", xlabel="episodes", ylabel="error in W", ylim=None)

compareplot(episodes, [VB, QB, probB],
            ["V", "Q", "Prob"], ["b", "orange", "k"],
            title="Random_MDP(50,4,46) alpha decay 10", xlabel="episodes", ylabel="error in B", ylim=None)



