import numpy as np
from copy import deepcopy
import random
from functions import rand_argmax
from agents import policy_agent
from wrapper import partial_env_A
from functions import one_hot


def Tabular_q(env, episodes, num_act, episode_length=np.inf, epsilon=0.05,
              alpha=lambda v, t: 0.1, gamma=0.99, eval_interval=np.inf, Qs=None, init=0, soft_end=False,
              Q_trafo=lambda x: x):
    # Q-lerning. Returns Q-values as a dict.
    # Alpha is a map from visit count and the elapsed time to the learning rate. eval_interval determines
    # after how many episodes the greedy policy is evaluated and the return printed. Qs allows for the initialization
    # of Q-values with a dictionary. If Qs is None, init allows for constant initialization at the value init.
    # soft end determines, how terminal states are treated. If soft_end is true, transitions to terminal states still
    # update on the Q-value of the next state. Q_trafo is the scalarization function that determines the action
    # selection in the multi-objective case.
    vs = {}
    if Qs is None:
        Qs = {}
    else:
        Qs = deepcopy(Qs)
    for i in range(episodes):
        obs_new = env.reset()
        if obs_new not in Qs.keys():
            Qs[obs_new] = [init for i in range(num_act)]
        if obs_new not in vs.keys():
            vs[obs_new] = [init for i in range(num_act)]
        done = False
        t = 0
        while done is False and t < episode_length:
            if obs_new not in Qs.keys():
                Qs[obs_new] = [init for i in range(num_act)]
            if obs_new not in vs.keys():
                vs[obs_new] = [init for i in range(num_act)]

            if np.random.uniform() > epsilon:
                act_new = rand_argmax(Q_trafo(Qs[obs_new]))
            else:
                act_new = np.random.choice(np.arange(num_act))

            if t > 0:
                error = (rew + gamma * Qs[obs_new][np.argmax(Q_trafo(Qs[obs_new]))] - Qs[obs][act])
                Qs[obs][act] = Qs[obs][act] + alpha(vs[obs][act], t) * error
                vs[obs][act] = vs[obs][act] + 1

            obs = obs_new
            act = act_new

            if hasattr(env, 'dynamic') and env.dynamic is True:
                obs_new, rew, done, _ = env.step(act, Qs)
            else:
                obs_new, rew, done, _ = env.step(act)

            if done is True:
                if soft_end is False:
                    error = (rew - Qs[obs][act])
                    Qs[obs][act] = Qs[obs][act] + alpha(vs[obs][act], t) * error
                    vs[obs][act] = vs[obs][act] + 1
                else:
                    if obs_new not in Qs.keys():
                        Qs[obs_new] = [init for i in range(num_act)]
                    error = (rew + gamma * Qs[obs_new][np.argmax(Q_trafo(Qs[obs_new]))] - Qs[obs][act])
                    Qs[obs][act] = Qs[obs][act] + alpha(vs[obs][act], t) * error
                    vs[obs][act] = vs[obs][act] + 1

            t = t + 1

        if i % eval_interval == (-1) % eval_interval:
            obs = env.reset()
            total = 0
            if obs not in Qs.keys():
                Qs[obs] = [init for i in range(num_act)]
            done = False
            s = 0
            while done is False and s < episode_length:
                act = rand_argmax(Q_trafo(Qs[obs]))
                obs_new, rew, done, _ = env.step(act)
                if obs_new not in Qs.keys():
                    Qs[obs_new] = [init for i in range(num_act)]
                obs = obs_new
                total = total + rew
                s = s + 1
            print(i, total)
    return Qs


def SARSA(env, episodes, num_act, policy_agent, episode_length=np.inf, epsilon=0.05,
          alpha=lambda v, t: 0.1, gamma=0.99, Qs=None, init=0, soft_end=False):
    # SARSA. Returns Q-values as a dict.
    # Alpha is a map from visit count and the elapsed time to the learning rate. Qs allows for the initialization
    # of Q-values with a dictionary. If Qs is None, init allows for constant initialization at the value init.
    # soft end determines, how terminal states are treated. If soft_end is true, transitions to terminal states still
    # update on the Q-value of the next state.
    vs = {}
    if Qs is None:
        Qs = {}
    else:
        Qs = deepcopy(Qs)
    for i in range(episodes):
        obs_new = env.reset()
        done = False
        t = 0
        while done is False and t < episode_length:
            if obs_new not in Qs.keys():
                Qs[obs_new] = [init for i in range(num_act)]
            if obs_new not in vs.keys():
                vs[obs_new] = [0 for i in range(num_act)]

            resample = False

            if np.random.uniform() > epsilon:
                act_new = policy_agent.act(obs_new)
            else:
                act_new = np.random.choice(np.arange(num_act))
                resample = True

            if t > 0:
                if resample == True:
                    act_target = policy_agent.act(obs_new)
                else:
                    act_target = act_new

                error = (rew + gamma * Qs[obs_new][act_target] - Qs[obs][act])
                Qs[obs][act] = Qs[obs][act] + alpha(vs[obs][act], t) * error
                vs[obs][act] = vs[obs][act] + 1

            obs = obs_new
            act = act_new
            obs_new, rew, done, _ = env.step(act)
            if done is True:
                if soft_end is False:
                    error = (rew - Qs[obs][act])
                    Qs[obs][act] = Qs[obs][act] + alpha(vs[obs][act], t) * error
                    vs[obs][act] = vs[obs][act] + 1
                else:
                    act_target = policy_agent.act(obs_new)
                    error = (rew + gamma * Qs[obs_new][act_target] - Qs[obs][act])
                    Qs[obs][act] = Qs[obs][act] + alpha(vs[obs][act], t) * error
                    vs[obs][act] = vs[obs][act] + 1
            t = t + 1
    return Qs


def Tdzero(env, episodes, num_act, policy_agent, episode_length=np.inf, epsilon=0.05,
           alpha=lambda v, t: 0.1, gamma=0.99, Vs=None, init=0, soft_end=False):
    # Td(0). Returns V-values as a dict.
    # Alpha is a map from visit count and the elapsed time to the learning rate. Vs allows for the initialization
    # of V-values with a dictionary. If Vs is None, init allows for constant initialization at the value init.
    # soft end determines, how terminal states are treated. If soft_end is true, transitions to terminal states still
    # update on the Q-value of the next state.
    vs = {}
    if Vs is None:
        Vs = {}
    else:
        Vs = deepcopy(Vs)
    for i in range(episodes):
        obs_new = env.reset()
        done = False
        resample = False
        t = 0
        while done is False and t < episode_length:
            if obs_new not in Vs.keys():
                Vs[obs_new] = init
            if obs_new not in vs.keys():
                vs[obs_new] = 0

            if t > 0:
                if resample is False:
                    error = (rew + gamma * Vs[obs_new] - Vs[obs])
                    Vs[obs] = Vs[obs] + alpha(vs[obs], t) * error
                    vs[obs] = vs[obs] + 1
                else:
                    resample = False

            if np.random.uniform() > epsilon:
                act_new = policy_agent.act(obs_new)
            else:
                act_new = np.random.choice(np.arange(num_act))
                resample = True

            obs = obs_new
            act = act_new

            obs_new, rew, done, _ = env.step(act)

            if done is True:
                if soft_end is False:
                    error = (rew - Vs[obs])
                    Vs[obs] = Vs[obs] + alpha(vs[obs], t) * error
                    vs[obs] = vs[obs] + 1
                else:
                    error = (rew + gamma * Vs[obs_new] - Vs[obs][act])
                    Vs[obs] = Vs[obs] + alpha(vs[obs], t) * error
                    vs[obs] = vs[obs] + 1
            t = t + 1
    return Vs


def get_F(env, episodes, num_act, policy, entries, exits, c=1, episode_length=np.inf, epsilon=0.05,
          alpha=lambda x, y: 0.1, gamma=0.99, evaluation="SARSA", re_evaluation_factor=0.25):
    # Returns the affinely linear operator F (the local option model) for a policy on a subset of the state space
    # defined by its entry and exits states under that policy, using the indirect method.
    # If evaluation is "SARSA", SARSA is used to approximate F, else td(0). c determines the bonus reward for exiting
    # that is used to calculate F. The larger c, the more accurate the method. re_evaluation_factor specifies, how many
    # episodes are to spend on learning with bonus rewards after the initialization without bonus rewards.

    if isinstance(next(iter(policy.values())), int):
        policy = {key: one_hot(policy[key], num_act) for key in policy}
        # entries to A, exits out of A
    B = np.zeros(len(entries))
    W = np.zeros((len(entries), len(exits)))
    x = np.zeros(len(exits))

    env_x = partial_env_A(env, entries, exits, values=x, gamma=gamma)

    if evaluation is "SARSA":
        Qs = SARSA(env_x, episodes, num_act, policy_agent(policy, num_act), episode_length=episode_length,
                   epsilon=epsilon,
                   alpha=alpha, gamma=gamma)
        for i, entry in enumerate(entries):
            Q = np.array(Qs[entry])
            V = np.average(Q, weights=policy[entry], axis=0)
            B[i] = V
    else:
        Vs = Tdzero(env_x, episodes, num_act, policy_agent(policy, num_act), episode_length=episode_length,
                    epsilon=epsilon,
                    alpha=alpha, gamma=gamma)
        for i, entry in enumerate(entries):
            V = Vs[entry]
            B[i] = V

    for k in range(len(exits)):
        x[k] = c
        x[:k] = 0
        env_x = partial_env_A(env, entries, exits, values=x, gamma=gamma)
        if evaluation is "SARSA":
            Qs_new = SARSA(env_x, int(episodes * re_evaluation_factor), num_act, policy_agent(policy, num_act),
                       episode_length=episode_length, epsilon=epsilon,
                       alpha=alpha, gamma=gamma, Qs=Qs)
            for i, entry in enumerate(entries):
                Q = np.array(Qs_new[entry])
                V = np.average(Q, weights=policy[entry], axis=0)
                V_pred = B[i]
                W[i, k] = (V - V_pred) / c
            
        else:
            Vs_new = Tdzero(env_x, int(episodes * re_evaluation_factor), num_act, policy_agent(policy, num_act),
                        episode_length=episode_length, epsilon=epsilon,
                        alpha=alpha, gamma=gamma, Vs=Vs)
            for i, entry in enumerate(entries):
                V = Vs_new[entry]
                V_pred = B[i]
                W[i, k] = (V - V_pred) / c

    return lambda v, i: B[i] + sum([v[j] * W[i, j] for j in range(len(v))]), W, B


def nested_key_else_zero(d, i, j):
    try:
        return d[i][j]
    except:
        return 0


def learn_matrices(env, episodes, policy_agent, entries, exits, episode_length=np.inf, gamma=0.99):
    # Returns the affinely linear operator F (the local option model) for a policy on a subset of the state space
    # defined by its entry and exits states under that policy, by approximating a transition model and direct
    # calculation.
    env_x = partial_env_A(env, entries, exits, values=np.zeros(len(exits)))
    reward_dict = {}
    transition_dict = {}
    transition_dict_terminal = {}
    for i in range(episodes):
        obs = env.reset()
        done = False
        t = 0
        while done is False and t < episode_length:
            t = t + 1
            act = policy_agent.act(obs)
            obs_new, rew, done, _ = env_x.step(act)
            try:
                s = sum([transition_dict[obs][key] for key in transition_dict[obs]])
                reward_dict[obs] = rew / s + (s - 1) * reward_dict[obs] / s
            except:
                reward_dict[obs] = rew
            if done is False:
                try:
                    transition_dict[obs][obs_new] = transition_dict[obs][obs_new] + 1
                except:
                    try:
                        transition_dict[obs][obs_new] = 1
                    except:
                        transition_dict[obs] = {}
                        transition_dict[obs][obs_new] = 1
            else:
                try:
                    transition_dict_terminal[obs][obs_new] = transition_dict_terminal[obs][obs_new] + 1
                except:
                    try:
                        transition_dict_terminal[obs][obs_new] = 1
                    except:
                        transition_dict_terminal[obs] = {}
                        transition_dict_terminal[obs][obs_new] = 1

            obs = obs_new

    index_array = {i: j for i, j in enumerate(reward_dict)}
    R = np.array([reward_dict[index_array[i]] for i in range(len(index_array))])

    P = []
    E = []
    for j in range(len(index_array)):
        e = [nested_key_else_zero(transition_dict_terminal, index_array[j], exits[i]) for i in range(len(exits))]
        p = [nested_key_else_zero(transition_dict, index_array[j], index_array[i]) for i in range(len(index_array))]
        z = (sum(p) + sum(e))
        p = np.array(p) / z
        e = np.array(e) / z
        P.append(p)
        E.append(e)
    P = np.array(P)
    E = np.array(E)

    B = np.matmul(np.linalg.inv((np.eye(P.shape[0]) - gamma * P)), R)
    W = np.matmul(np.linalg.inv((np.eye(P.shape[0]) - gamma * P)), gamma * E)

    indexes = []
    for key in index_array:
        if index_array[key] in entries:
            indexes.append(key)
    indexes.sort()

    indexes = np.array(indexes)
    B = B[indexes]
    W = W[indexes]

    # selection!!!

    return lambda v, i: B[i] + sum([v[j] * W[i, j] for j in range(len(v))]), W, B













