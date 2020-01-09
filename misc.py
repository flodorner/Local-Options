import numpy as np
import scipy.stats as stat
from matplotlib import pyplot as plt


def mean_confidence_interval(data, confidence=0.95):
    # Computes confidence interval (assuming data is normally distributed)
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stat.sem(a)
    h = se * stat.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def compareplot(axis, arrays, names, colors, title="", xlabel="x",
                ylabel="y", ylim=(0, 10), save=False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    for i, array in enumerate(arrays):
        means, upper, lower = [], [], []
        for item in array:
            m, mp, mm = mean_confidence_interval(item)
            means.append(m)
            upper.append(mm)
            lower.append(mp)
        ax.plot(axis, means, colors[i])
        ax.fill_between(axis, lower, upper, facecolor=colors[i], interpolate=True, alpha=0.2, label="_nolegend_")
    ax.legend(names)
    if save is False:
        plt.show()
    else:
        plt.savefig(save)