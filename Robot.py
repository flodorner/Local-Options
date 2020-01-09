import numpy as np
from matplotlib import pyplot as plt

from agents import policy_agent_deterministic
from algorithms import Tabular_q, SARSA, learn_matrices, Tdzero
from environments import robot
from functions import Multi_Qs_to_F, Qs_to_policy
from misc import compareplot
from wrapper import partial_env_A_multireward, partial_env_B

N = 100
epsnum = [10 ** 2, int(10 ** 2.5), 10 ** 3, int(10 ** 3.5), int(10 ** 4), int(10 ** 4.5),
          int(10 ** 5)]

def alpha(v, t):
    return 10 / (v + 10)


Q0 = []
Q1 = []
Q2 = []
Q3 = []
epslen = []

for e in epsnum:
    Q0_temp = []
    Q1_temp = []
    Q2_temp = []
    Q3_temp = []
    epslen_temp = []
    for k in range(N):
        env = robot()
        Qs = Tabular_q(env, e, 4, episode_length=20, epsilon=0.25,
                       alpha=alpha, gamma=0.9, eval_interval=np.inf, soft_end=True)
        start_values = Qs[(16, 0, 0)]
        Q0_temp.append(start_values[0])
        Q1_temp.append(start_values[1])
        Q2_temp.append(start_values[2])
        Q3_temp.append(start_values[3])
        epslen_temp.append(env.steps)
    Q0.append(Q0_temp)
    Q1.append(Q1_temp)
    Q2.append(Q2_temp)
    Q3.append(Q3_temp)
    epslen.append(epslen_temp)

compareplot(np.log10(epsnum), [Q0, Q1, Q2, Q3],
            ["Action 0", "Action 1", "Action 2", "Action 3"], ["b", "orange", "red", "black"],
            title="Q-learning for the Robot MDP with gamma=0.9", xlabel="log 10 episodes", ylabel="Q-value",
            ylim=(0, 150))
compareplot(np.log10(epsnum), [epslen],
            ["Interactions"], ["b"],
            title="Number of interactions with the environment", xlabel="log 10 episodes",
            ylabel="Interactions", ylim=(0, 201000 * 10))
print(Q2)

env = robot()


def alpha(v, t):
    return 10 / (v + 10)


a = partial_env_A_multireward(env, [(-i, 0, 0) for i in range(17)], [(i + 1, 0, 0) for i in range(16)], gamma=0.9)

policy_dict = {(0, 0, 0): 3, (-1, 0, 0): 3, (-2, 0, 0): 3, (-3, 0, 0): 3, (-4, 0, 0): 3, (-5, 0, 0): 3, (-6, 0, 0): 3,
               (-7, 0, 0): 3, (-8, 0, 0): 3, (-9, 0, 0): 3, (-10, 0, 0): 3, (-11, 0, 0): 3, (-12, 0, 0): 3,
               (-13, 0, 0): 3,
               (-14, 0, 0): 3, (-15, 0, 0): 3, (-16, 0, 0): 1}
policy = policy_agent_deterministic(policy_dict, 4)

Qs = SARSA(a, 1000, 4, policy, episode_length=5, epsilon=0, alpha=alpha, gamma=0.9)
base = a.env.steps
F, W, B = Multi_Qs_to_F(policy_dict, Qs, [(-i, 0, 0) for i in range(17)])

b = partial_env_B(env, [(i + 1, 0, 0) for i in range(16)], [(-i, 0, 0) for i in range(17)], F, 4, gamma=0.9,
                  entry_distribution=None)

epsnum = [10 ** 2, int(10 ** 2.5), 10 ** 3, int(10 ** 3.5), int(10 ** 4), int(10 ** 4.5),
          int(10 ** 5)]


def alpha(v, t):
    return 10 / (v + 10)


Q0 = []
Q1 = []
Q2 = []
Q3 = []
epslen = []

for e in epsnum:
    Q0_temp = []
    Q1_temp = []
    Q2_temp = []
    Q3_temp = []
    epslen_temp = []
    for k in range(N):
        env = robot()
        b = partial_env_B(env, [(i + 1, 0, 0) for i in range(16)], [(-i, 0, 0) for i in range(17)], F, 4, gamma=0.9,
                          entry_distribution=None)
        Qs = Tabular_q(b, e, 4, episode_length=20, epsilon=0.25,
                       alpha=alpha, gamma=0.9, eval_interval=np.inf)
        epslen_temp.append(b.env.steps)
        start_values = Qs[(16, 0, 0)]
        Q0_temp.append(start_values[0])
        Q1_temp.append(start_values[1])
        Q2_temp.append(start_values[2])
        Q3_temp.append(start_values[3])
    Q0.append(Q0_temp)
    Q1.append(Q1_temp)
    Q2.append(Q2_temp)
    Q3.append(Q3_temp)
    epslen.append(epslen_temp)

compareplot(np.log10(epsnum), [Q0, Q1, Q2, Q3],
            ["Action 0", "Action 1", "Action 2", "Action 3"], ["b", "orange", "red", "black"],
            title="Expert Q-learning for the Robot MDP with gamma=0.9", xlabel="log 10 episodes", ylabel="Q-value",
            ylim=(0, 150))

compareplot(np.log10(epsnum), [epslen, np.array(epslen) + base],
            ["Without pretraining", "With pretraining"], ["b", "navy"],
            title="Number of interactions with the environment", xlabel="log 10 episodes",
            ylabel="Interactions", ylim=(0, 201000 * 10))
print(Q2)

env = robot()


def alpha(v, t):
    return 10 / (v + 10)


a = partial_env_A_multireward(env, [(-i, 0, 0) for i in range(17)], [(i + 1, 0, 0) for i in range(16)], gamma=0.9)

policy_dict = {(0, 0, 0): 3, (-1, 0, 0): 3, (-2, 0, 0): 3, (-3, 0, 0): 3, (-4, 0, 0): 3, (-5, 0, 0): 3, (-6, 0, 0): 3,
               (-7, 0, 0): 3, (-8, 0, 0): 3, (-9, 0, 0): 3, (-10, 0, 0): 3, (-11, 0, 0): 3, (-12, 0, 0): 3,
               (-13, 0, 0): 3,
               (-14, 0, 0): 3, (-15, 0, 0): 3, (-16, 0, 0): 1}
policy = policy_agent_deterministic(policy_dict, 4)

Qs = SARSA(a, 1000, 4, policy, episode_length=5, epsilon=0, alpha=alpha, gamma=0.9)
F1, W1, B1 = Multi_Qs_to_F(policy_dict, Qs, [(-i, 0, 0) for i in range(17)])
base = a.env.steps
policy_dict = {(0, 0, 0): 3, (-1, 0, 0): 3, (-2, 0, 0): 3, (-3, 0, 0): 3, (-4, 0, 0): 3, (-5, 0, 0): 3, (-6, 0, 0): 3,
               (-7, 0, 0): 3, (-8, 0, 0): 3, (-9, 0, 0): 3, (-10, 0, 0): 3, (-11, 0, 0): 3, (-12, 0, 0): 1,
               (-13, 0, 0): 1,
               (-14, 0, 0): 1, (-15, 0, 0): 1, (-16, 0, 0): 1}
policy = policy_agent_deterministic(policy_dict, 4)

Qs = SARSA(a, 1000, 4, policy, episode_length=5, epsilon=0, alpha=alpha, gamma=0.9)
F2, W2, B2 = Multi_Qs_to_F(policy_dict, Qs, [(-i, 0, 0) for i in range(17)])
base += a.env.steps
policy_dict = {(0, 0, 0): 3, (-1, 0, 0): 3, (-2, 0, 0): 3, (-3, 0, 0): 3, (-4, 0, 0): 3, (-5, 0, 0): 3, (-6, 0, 0): 3,
               (-7, 0, 0): 3, (-8, 0, 0): 3, (-9, 0, 0): 1, (-10, 0, 0): 1, (-11, 0, 0): 1, (-12, 0, 0): 1,
               (-13, 0, 0): 1,
               (-14, 0, 0): 1, (-15, 0, 0): 1, (-16, 0, 0): 1}
policy = policy_agent_deterministic(policy_dict, 4)
Qs = SARSA(a, 1000, 4, policy, episode_length=5, epsilon=0, alpha=alpha, gamma=0.9)
F3, W3, B3 = Multi_Qs_to_F(policy_dict, Qs, [(-i, 0, 0) for i in range(17)])
base += a.env.steps


def F(v, i):
    return max([F1(v, i), F2(v, i), F3(v, i)])


b = partial_env_B(env, [(i + 1, 0, 0) for i in range(16)], [(-i, 0, 0) for i in range(17)], F, 4, gamma=0.9,
                  entry_distribution=None)

epsnum = [10 ** 2, int(10 ** 2.5), 10 ** 3, int(10 ** 3.5), int(10 ** 4), int(10 ** 4.5),
          int(10 ** 5)]


def alpha(v, t):
    return 10 / (v + 10)


Q0 = []
Q1 = []
Q2 = []
Q3 = []
epslen = []

for e in epsnum:
    Q0_temp = []
    Q1_temp = []
    Q2_temp = []
    Q3_temp = []
    epslen_temp = []
    for k in range(N):
        env = robot()
        b = partial_env_B(env, [(i + 1, 0, 0) for i in range(16)], [(-i, 0, 0) for i in range(17)], F, 4, gamma=0.9,
                          entry_distribution=None)
        Qs = Tabular_q(b, e, 4, episode_length=20, epsilon=0.25,
                       alpha=alpha, gamma=0.9, eval_interval=np.inf)
        start_values = Qs[(16, 0, 0)]
        Q0_temp.append(start_values[0])
        Q1_temp.append(start_values[1])
        Q2_temp.append(start_values[2])
        Q3_temp.append(start_values[3])
        epslen_temp.append(b.env.steps)
    Q0.append(Q0_temp)
    Q1.append(Q1_temp)
    Q2.append(Q2_temp)
    Q3.append(Q3_temp)
    epslen.append(epslen_temp)

compareplot(np.log10(epsnum), [Q0, Q1, Q2, Q3],
            ["Action 0", "Action 1", "Action 2", "Action 3"], ["b", "orange", "red", "black"],
            title="Multiexpert Q-learning for the Robot MDP with gamma=0.9", xlabel="log 10 episodes", ylabel="Q-value",
            ylim=(0, 150))
compareplot(np.log10(epsnum), [epslen, np.array(epslen) + base],
            ["Without pretraining", "With pretraining"], ["b", "navy"],
            title="Number of interactions with the environment", xlabel="log 10 episodes",
            ylabel="Interactions", ylim=(0, 201000 * 10))

print(Q2)

env = robot()


def alpha(v, t):
    return 10 / (v + 10)


a = partial_env_A_multireward(env, [(-i, 0, 0) for i in range(17)], [(i + 1, 0, 0) for i in range(16)], gamma=0.9)

policy_dict = {(0, 0, 0): 3, (-1, 0, 0): 3, (-2, 0, 0): 3, (-3, 0, 0): 3, (-4, 0, 0): 3, (-5, 0, 0): 3, (-6, 0, 0): 3,
               (-7, 0, 0): 3, (-8, 0, 0): 3, (-9, 0, 0): 3, (-10, 0, 0): 3, (-11, 0, 0): 3, (-12, 0, 0): 3,
               (-13, 0, 0): 3,
               (-14, 0, 0): 1, (-15, 0, 0): 1, (-16, 0, 0): 1}
policy = policy_agent_deterministic(policy_dict, 4)

Qs = SARSA(a, 1000, 4, policy, episode_length=5, epsilon=0, alpha=alpha, gamma=0.9)
F, W, B = Multi_Qs_to_F(policy_dict, Qs, [(-i, 0, 0) for i in range(17)])

b = partial_env_B(env, [(i + 1, 0, 0) for i in range(16)], [(-i, 0, 0) for i in range(17)], F, 4, gamma=0.9,
                  entry_distribution=None)

epsnum = [10 ** 2, int(10 ** 2.5), 10 ** 3, int(10 ** 3.5), int(10 ** 4), int(10 ** 4.5),
          int(10 ** 5)]


def alpha(v, t):
    return 10 / (v + 10)


Q0 = []
Q1 = []
Q2 = []
Q3 = []

for e in epsnum:
    Q0_temp = []
    Q1_temp = []
    Q2_temp = []
    Q3_temp = []
    for k in range(N):
        Qs = Tabular_q(b, e, 4, episode_length=20, epsilon=0.25,
                       alpha=alpha, gamma=0.9, eval_interval=np.inf)
        start_values = Qs[(16, 0, 0)]
        Q0_temp.append(start_values[0])
        Q1_temp.append(start_values[1])
        Q2_temp.append(start_values[2])
        Q3_temp.append(start_values[3])
    Q0.append(Q0_temp)
    Q1.append(Q1_temp)
    Q2.append(Q2_temp)
    Q3.append(Q3_temp)

compareplot(np.log10(epsnum), [Q0, Q1, Q2, Q3],
            ["Action 0", "Action 1", "Action 2", "Action 3"], ["b", "orange", "red", "black"],
            title="Expert Q-learning with suboptimal expert", xlabel="log 10 episodes", ylabel="Q-value", ylim=(0, 150))
print(Q2)

env = robot()


def alpha(v, t):
    return 10 / (v + 10)


a = partial_env_A_multireward(env, [(-i, 0, 0) for i in range(17)], [(i + 1, 0, 0) for i in range(16)], gamma=0.9)

policy_dict = {(0, 0, 0): 3, (-1, 0, 0): 3, (-2, 0, 0): 3, (-3, 0, 0): 3, (-4, 0, 0): 3, (-5, 0, 0): 3, (-6, 0, 0): 3,
               (-7, 0, 0): 3, (-8, 0, 0): 3, (-9, 0, 0): 3, (-10, 0, 0): 3, (-11, 0, 0): 3, (-12, 0, 0): 3,
               (-13, 0, 0): 3,
               (-14, 0, 0): 3, (-15, 0, 0): 3, (-16, 0, 0): 1}
policy = policy_agent_deterministic(policy_dict, 4)

Qs_Sarsa = SARSA(a, 100000, 4, policy, episode_length=5, epsilon=0, alpha=alpha, gamma=0.9)

noise = [1e-4, 1e-3, 1e-2, 1e-1, 10 ** (-0.9), 10 ** (-0.8), 10 ** (-0.7), 10 ** (-0.6), 10 ** (-0.5)]

Q0 = []
Q1 = []
Q2 = []
Q3 = []

for std in noise:
    Qs_noisy = {
        key: [(Q + np.random.normal(scale=std, size=len(Q)) if type(Q) == np.ndarray else 0) for Q in Qs_Sarsa[key]] for
    key
        in Qs_Sarsa}
    F, W, B = Multi_Qs_to_F(policy_dict, Qs_noisy, [(-i, 0, 0) for i in range(17)])
    b = partial_env_B(env, [(i + 1, 0, 0) for i in range(16)], [(-i, 0, 0) for i in range(17)], F, 4, gamma=0.9,
                      entry_distribution=None)
    Q0_temp = []
    Q1_temp = []
    Q2_temp = []
    Q3_temp = []
    for k in range(N):
        Qs = Tabular_q(b, 1000, 4, episode_length=10, epsilon=0.25,
                       alpha=alpha, gamma=0.9, eval_interval=np.inf)
        start_values = Qs[(16, 0, 0)]
        Q0_temp.append(start_values[0])
        Q1_temp.append(start_values[1])
        Q2_temp.append(start_values[2])
        Q3_temp.append(start_values[3])
    Q0.append(Q0_temp)
    Q1.append(Q1_temp)
    Q2.append(Q2_temp)
    Q3.append(Q3_temp)

compareplot(np.log10(noise), [Q0, Q1, Q2, Q3],
            ["Action 0", "Action 1", "Action 2", "Action 3"], ["b", "orange", "red", "black"],
            title="Expert Q-learning for the Robot MDP with perturbed F", xlabel="log 10 std", ylabel="Q-value",
            ylim=(0, 150))
print(Q2)


def alpha(v, t):
    return 10 / (v + 10)


env = robot()


def exit(vector):
    return vector[0] + vector[-1]


def Q_trafo(Qs):
    Q = [exit(Qs[i]) if isinstance(Qs[i], np.ndarray) else 0 for i in range(len(Qs))]
    return Q


policy_dict = {(0, 0, 0): 3, (-1, 0, 0): 3, (-2, 0, 0): 3, (-3, 0, 0): 3, (-4, 0, 0): 3, (-5, 0, 0): 3, (-6, 0, 0): 3,
               (-7, 0, 0): 3, (-8, 0, 0): 3, (-9, 0, 0): 3, (-10, 0, 0): 3, (-11, 0, 0): 3, (-12, 0, 0): 3,
               (-13, 0, 0): 3,
               (-14, 0, 0): 3, (-15, 0, 0): 3, (-16, 0, 0): 1}
policy = policy_agent_deterministic(policy_dict, 4)

a = partial_env_A_multireward(env, [(-i, 0, 0) for i in range(17)], [(i + 1, 0, 0) for i in range(16)], gamma=0.9)
Qs = SARSA(a, 1000, 4, policy, episode_length=100, epsilon=0, alpha=alpha, gamma=0.9)
F, W, B = Multi_Qs_to_F(policy_dict, Qs, [(-i, 0, 0) for i in range(17)])

epsnum = [10 ** 1, int(10 ** 1.5), 10 ** 2, int(10 ** 2.5), 10 ** 3, int(10 ** 3.5)]
epslens = []
Diffs = []
for e in epsnum:
    Diff = []
    epslen = []
    for i in range(int(N / 4)):
        env = robot()
        a = partial_env_A_multireward(env, [(-i, 0, 0) for i in range(17)], [(i + 1, 0, 0) for i in range(16)],
                                      gamma=0.9)
        Qs = Tabular_q(a, e, 4, episode_length=20, epsilon=0.25, alpha=alpha, gamma=0.9, Q_trafo=Q_trafo)
        base = a.env.steps
        epslen.append(base)
        policy = Qs_to_policy(Qs, Q_trafo)
        policy_dict = {}
        for j in [(-i, 0, 0) for i in range(17)]:
            try:
                policy_dict[j] = policy[j]
            except:
                policy_dict[j] = 0
        F1, W1, B1 = Multi_Qs_to_F(policy_dict, Qs, [(-i, 0, 0) for i in range(17)])
        Diff.append(np.linalg.norm(W - W1))

    epslens.append(epslen)
    Diffs.append(Diff)

compareplot(np.log10(epsnum), [Diffs],
            ["Difference in W"], ["b"],
            title="Q-learning vs SARSA for approximating F", xlabel="log 10 episodes", ylabel="Error", ylim=(0, 10))
compareplot(np.log10(epsnum), [epslens],
            ["Q-based pretraining"], ["b"],
            title="Number of interactions with the environment", xlabel="log 10 episodes",
            ylabel="Interactions", ylim=(0, 201000 * 10))


def alpha(v, t):
    return 10 / (v + 10)


env = robot()


def exit(vector):
    return vector[0] + vector[-1]


def Q_trafo(Qs):
    Q = [exit(Qs[i]) if isinstance(Qs[i], np.ndarray) else 0 for i in range(len(Qs))]
    return Q


policy_dict = {(0, 0, 0): 3, (-1, 0, 0): 3, (-2, 0, 0): 3, (-3, 0, 0): 3, (-4, 0, 0): 3, (-5, 0, 0): 3, (-6, 0, 0): 3,
               (-7, 0, 0): 3, (-8, 0, 0): 3, (-9, 0, 0): 3, (-10, 0, 0): 3, (-11, 0, 0): 3, (-12, 0, 0): 3,
               (-13, 0, 0): 3,
               (-14, 0, 0): 3, (-15, 0, 0): 3, (-16, 0, 0): 1}
policy = policy_agent_deterministic(policy_dict, 4)

a = partial_env_A_multireward(env, [(-i, 0, 0) for i in range(17)], [(i + 1, 0, 0) for i in range(16)], gamma=0.9)
Qs = SARSA(a, 1000, 4, policy, episode_length=100, epsilon=0, alpha=alpha, gamma=0.9)
F, W, B = Multi_Qs_to_F(policy_dict, Qs, [(-i, 0, 0) for i in range(17)])

epsnum = [10 ** 1, int(10 ** 1.5), 10 ** 2, int(10 ** 2.5), 10 ** 3, int(10 ** 3.5)]
Diffs = []
for e in epsnum:
    Diff = []
    for i in range(int(N / 4)):
        a = partial_env_A_multireward(env, [(-i, 0, 0) for i in range(17)], [(i + 1, 0, 0) for i in range(16)],
                                      gamma=0.9)
        Qs = SARSA(a, e, 4, policy, episode_length=5, epsilon=0, alpha=alpha, gamma=0.9)
        F1, W1, B1 = Multi_Qs_to_F(policy_dict, Qs, [(-i, 0, 0) for i in range(17)])
        Diff.append(np.linalg.norm(W - W1))

    Diffs.append(Diff)

compareplot(np.log10(epsnum), [Diffs],
            ["Difference in W"], ["b"],
            title="SARSA vs SARSA for approximating F", xlabel="log 10 episodes", ylabel="Error", ylim=(0, 10))

eps = [0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]


def alpha(v, t):
    return 10 / (v + 10)


Q0 = []
Q1 = []
Q2 = []
Q3 = []

for e in eps:
    Q0_temp = []
    Q1_temp = []
    Q2_temp = []
    Q3_temp = []
    for k in range(N):
        env = robot()
        Qs = Tabular_q(env, 10 ** 3, 4, episode_length=20, epsilon=e,
                       alpha=alpha, gamma=0.9, eval_interval=np.inf, soft_end=True)
        start_values = Qs[(16, 0, 0)]
        Q0_temp.append(start_values[0])
        Q1_temp.append(start_values[1])
        Q2_temp.append(start_values[2])
        Q3_temp.append(start_values[3])
    Q0.append(Q0_temp)
    Q1.append(Q1_temp)
    Q2.append(Q2_temp)
    Q3.append(Q3_temp)

compareplot(eps, [Q0, Q1, Q2, Q3],
            ["Action 0", "Action 1", "Action 2", "Action 3"], ["b", "orange", "red", "black"],
            title="Q-learning for the Robot MDP with gamma=0.9", xlabel="epsilon", ylabel="Q-value", ylim=(0, 150))

print(Q2)

env = robot()


def alpha(v, t):
    return 10 / (v + 10)


a = partial_env_A_multireward(env, [(-i, 0, 0) for i in range(17)], [(i + 1, 0, 0) for i in range(16)], gamma=0.9)

policy_dict = {(0, 0, 0): 3, (-1, 0, 0): 3, (-2, 0, 0): 3, (-3, 0, 0): 3, (-4, 0, 0): 3, (-5, 0, 0): 3, (-6, 0, 0): 3,
               (-7, 0, 0): 3, (-8, 0, 0): 3, (-9, 0, 0): 3, (-10, 0, 0): 3, (-11, 0, 0): 3, (-12, 0, 0): 3,
               (-13, 0, 0): 3,
               (-14, 0, 0): 3, (-15, 0, 0): 3, (-16, 0, 0): 1}
policy = policy_agent_deterministic(policy_dict, 4)

Qs = SARSA(a, 1000, 4, policy, episode_length=20, epsilon=0, alpha=alpha, gamma=0.9)
F, W, B = Multi_Qs_to_F(policy_dict, Qs, [(-i, 0, 0) for i in range(17)])

b = partial_env_B(env, [(i + 1, 0, 0) for i in range(16)], [(-i, 0, 0) for i in range(17)], F, 4, gamma=0.9,
                  entry_distribution=None)

eps = [0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]


def alpha(v, t):
    return 10 / (v + 10)


Q0 = []
Q1 = []
Q2 = []
Q3 = []

for e in eps:
    Q0_temp = []
    Q1_temp = []
    Q2_temp = []
    Q3_temp = []
    for k in range(N):
        Qs = Tabular_q(b, 10 ** 3, 4, episode_length=20, epsilon=e,
                       alpha=alpha, gamma=0.9, eval_interval=np.inf)
        start_values = Qs[(16, 0, 0)]
        Q0_temp.append(start_values[0])
        Q1_temp.append(start_values[1])
        Q2_temp.append(start_values[2])
        Q3_temp.append(start_values[3])
    Q0.append(Q0_temp)
    Q1.append(Q1_temp)
    Q2.append(Q2_temp)
    Q3.append(Q3_temp)

compareplot(eps, [Q0, Q1, Q2, Q3],
            ["Action 0", "Action 1", "Action 2", "Action 3"], ["b", "orange", "red", "black"],
            title="Expert Q-learning for the Robot MDP with gamma=0.9", xlabel="epsilon", ylabel="Q-value",
            ylim=(0, 150))
print(Q2)

epslength = [2, 2 * int(10 ** (0.5)), 2 * int(10 ** 1), 2 * int(10 ** (1.5)), 2 * int(10 ** 2), 2 * int(10 ** 2.5),
             2 * int(10 ** 3)]


def alpha(v, t):
    return 10 / (v + 10)


Q0 = []
Q1 = []
Q2 = []
Q3 = []
epslen = []

for e in epslength:
    Q0_temp = []
    Q1_temp = []
    Q2_temp = []
    Q3_temp = []
    epslen_temp = []
    for k in range(N):
        env = robot()
        Qs = Tabular_q(env, 10 ** 3, 4, episode_length=e, epsilon=0.25,
                       alpha=alpha, gamma=0.9, eval_interval=np.inf, soft_end=True)
        start_values = Qs[(16, 0, 0)]
        Q0_temp.append(start_values[0])
        Q1_temp.append(start_values[1])
        Q2_temp.append(start_values[2])
        Q3_temp.append(start_values[3])
        epslen_temp.append(env.steps)
    Q0.append(Q0_temp)
    Q1.append(Q1_temp)
    Q2.append(Q2_temp)
    Q3.append(Q3_temp)
    epslen.append(epslen_temp)

compareplot(np.log10(np.array(epslength)), [Q0, Q1, Q2, Q3],
            ["Action 0", "Action 1", "Action 2", "Action 3"], ["b", "orange", "red", "black"],
            title="Q-learning for the Robot MDP with gamma=0.9", xlabel="log10 episode length", ylabel="Q-value",
            ylim=(0, 150))
compareplot(np.log10(np.array(epslength)), [epslen],
            ["Interactions"], ["b"],
            title="Number of interactions with the environment", xlabel="log 10 episode length",
            ylabel="Interactions", ylim=(0, 201000 * 10))

print(Q2)

env = robot()
total = 0
steps = 0
env.reset()
obs, rew, done, _ = env.step(2)
print(obs)
total += rew * 0.9 ** steps
steps += 1
obs, rew, done, _ = env.step(2)
print(obs)
total += rew * 0.9 ** steps
steps += 1
obs, rew, done, _ = env.step(2)
print(obs)
total += rew * 0.9 ** steps
steps += 1
for j in range(3000):
    obs, rew, done, _ = env.step(3)
    total += rew * 0.9 ** steps
    steps += 1
    obs, rew, done, _ = env.step(3)
    total += rew * 0.9 ** steps
    steps += 1
    obs, rew, done, _ = env.step(3)
    total += rew * 0.9 ** steps
    steps += 1
    obs, rew, done, _ = env.step(3)
    total += rew * 0.9 ** steps
    steps += 1
    obs, rew, done, _ = env.step(2)
    total += rew * 0.9 ** steps
    steps += 1
    obs, rew, done, _ = env.step(2)
    total += rew * 0.9 ** steps
    steps += 1
    obs, rew, done, _ = env.step(2)
    total += rew * 0.9 ** steps
    steps += 1
    obs, rew, done, _ = env.step(2)
    total += rew * 0.9 ** steps
    steps += 1
print(total)
