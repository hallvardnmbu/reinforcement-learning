"""Visualisation utilities for the Tetris environment and agents."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def plot(metrics, title, window=50):
    """
    Visualise the training metrics from a CSV file.

    Parameters
    ----------
    metrics : dict
    title : str
    window : int

    Returns
    -------
    plt.Figure
    """
    steps = gaussian_filter1d(metrics["steps"], sigma=window)
    losses = gaussian_filter1d(metrics["losses"], sigma=window)

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(title)

    ax[0].plot(steps, color="black", linewidth=1)
    ax[0].set_title("Average steps per game")
    ax[0].set_xticks([])

    ax[1].plot(np.linspace(0, len(metrics["steps"]), len(losses)), losses,
               color="black", linewidth=1)
    ax[1].set_title("Average loss")
    ax[1].set_xlabel("Game nr.")
    ax[1].set_yscale("log")

    axs = ax[1].twinx()
    axs.plot(metrics["exploration"], color="gray", linewidth=0.5)
    axs.tick_params(axis='y', colors='gray')
    axs.set_ylabel("Exploration rate")
    axs.yaxis.label.set_color('gray')

    return fig


def graph(path, title, window=50):
    """
    Visualise the training metrics from a CSV file.

    Parameters
    ----------
    path : str
    title : str
    window : int
        The window size for the rolling averages.

    Returns
    -------
    plt.Figure
    """
    metrics = pd.read_csv(path, header=0).set_index("game", drop=True)

    steps = gaussian_filter1d(metrics["steps"], sigma=window)
    rewards = metrics.loc[metrics["reward"] != 0, "reward"]

    loss = gaussian_filter1d(metrics["loss"].dropna(), sigma=window)
    exploration = metrics["exploration"].rolling(window=window, min_periods=1).mean()

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(title + f" (window size {window})")

    training = metrics[metrics['loss'] > 0].index[0]
    if training > metrics.shape[0] / 50:
        ax[0].axvline(x=training, color='black', linewidth=1)
        ax[1].axvline(x=training, color='black', linewidth=1)
        ax[1].text(training, 0.5, 'Training starts',
                   verticalalignment='center', horizontalalignment='right',
                   transform=ax[1].get_xaxis_transform(),
                   rotation=90)

    ax[0].plot(exploration, color='gray', linewidth=1)
    ax[0].set_yticks([i / 10 for i in range(0, 11, 2)])
    ax[0].set_ylabel("Exploration rate", color='gray')
    ax[0].tick_params(axis='y', colors='gray')
    ax[0].set_xlim(0, metrics.shape[0])
    ax[0].set_ylim(-0.1, 1.1)

    step = ax[0].twinx()
    step.plot(steps, color='black', linewidth=0.5)
    step.set_ylabel("Steps")
    step.set_xlim(0, metrics.shape[0])

    ax[1].scatter(rewards.index, rewards.values, marker='*', color='orange', s=25)
    ax[1].set_yticks(np.linspace(1, metrics["reward"].max(), 9)) \
        if metrics["reward"].max() > 1 else None
    ax[1].set_ylabel("\u2605 Reward", color='orange')
    ax[1].set_xlabel("Game nr.")
    ax[1].tick_params(axis='y', colors='orange')
    ax[1].set_xlim(0, metrics.shape[0])

    losses = ax[1].twinx()
    x_values = np.linspace(training, metrics.shape[0], len(loss))
    losses.plot(x_values, loss, color='black', linewidth=1)
    losses.tick_params(axis='y')
    losses.set_ylabel("Loss")
    losses.set_xlim(0, metrics.shape[0])
    losses.set_yscale("log")

    return fig


def graph2(path, title, window=50):
    """
    Visualise the training metrics from a CSV file.

    Parameters
    ----------
    path : str
    title : str
    window : int
        The window size for the rolling averages.

    Returns
    -------
    plt.Figure
    """
    metrics = pd.read_csv(path, header=0).set_index("game", drop=True)

    steps = gaussian_filter1d(metrics["steps"], sigma=window)
    rewards = metrics.loc[metrics["reward"] != 0, "reward"]

    loss = gaussian_filter1d(metrics["loss"].dropna(), sigma=window)
    exploration = metrics["exploration"].rolling(window=window, min_periods=1).mean()

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(title + f" (window size {window})")

    training = metrics[metrics['loss'] > 0].index[0]
    if training > metrics.shape[0] / 50:
        ax[0].axvline(x=training, color='black', linewidth=1)
        ax[1].axvline(x=training, color='black', linewidth=1)
        ax[1].text(training, 0.5, 'Training starts',
                   verticalalignment='center', horizontalalignment='right',
                   transform=ax[1].get_xaxis_transform(),
                   rotation=90)

    ax[0].plot(exploration, color='gray', linewidth=1)
    ax[0].set_yticks([i / 10 for i in range(0, 11, 2)])
    ax[0].set_ylabel("Exploration rate", color='gray')
    ax[0].tick_params(axis='y', colors='gray')
    ax[0].set_xlim(0, metrics.shape[0])
    ax[0].set_ylim(-0.1, 1.1)

    step = ax[0].twinx()
    step.plot(steps, color='black', linewidth=0.5)
    step.set_ylabel("Steps")
    step.set_xlim(0, metrics.shape[0])

    bins = pd.cut(rewards.index,
                  bins=range(0, metrics.shape[0], metrics.shape[0] // 10),
                  include_lowest=True)
    grouped_rewards = rewards.groupby([bins, rewards], observed=True).count()
    x_values = [interval.mid for interval in grouped_rewards.index.get_level_values(0)]
    y_values = grouped_rewards.index.get_level_values(1).tolist()
    marker_sizes = 25 + grouped_rewards.values * 2
    ax[1].scatter(x_values, y_values, s=marker_sizes, marker='*', color='orange')
    ax[1].set_yticks(list(set(metrics["reward"].unique()) - {0.0}))
    ax[1].set_ylabel("\u2605 Reward", color='orange')
    ax[1].set_xlabel("Game nr.")
    ax[1].tick_params(axis='y', colors='orange')
    ax[1].set_xlim(0, metrics.shape[0])

    losses = ax[1].twinx()
    x_values = np.linspace(training, metrics.shape[0], len(loss))
    losses.plot(x_values, loss, color='black', linewidth=1)
    losses.tick_params(axis='y')
    losses.set_ylabel("Loss")
    losses.set_xlim(0, metrics.shape[0])
    losses.set_yscale("log")

    return fig
