"""Visualisation utilities for the Tetris environment and agents."""

import torch
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _roll(data, window=50):
    half_window = window // 2
    return [(data[max(0, i - half_window):min(len(data), i + half_window)]).mean()
            for i in range(len(data))]


def visualise(metrics, title, window=150):
    """
    Visualise the training metrics from a CSV file.

    Parameters
    ----------
    metrics : dict
    title : str
    window : int
        The window size for the rolling averages.

    Returns
    -------
    plt.Figure
    """
    steps = _roll(metrics["steps"], window)
    losses = _roll(metrics["losses"], window)

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


def graph(path, title, window=150):
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

    steps = metrics["steps"].rolling(window=window, center=True).mean()
    rewards = metrics.loc[metrics["reward"] != 0, "reward"]

    loss = metrics["loss"].dropna().rolling(window=window, min_periods=1).mean()
    exploration = metrics["exploration"].rolling(window=window, min_periods=1).mean()

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(title + f" (window size {window})")

    training = metrics[metrics['loss'] > 0].index[0]
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
    ax[1].set_yticks(list(set(metrics["reward"].unique()) - {0.0}))
    ax[1].set_ylabel("\u2605 Reward", color='orange')
    ax[1].set_xlabel("Game nr.")
    ax[1].tick_params(axis='y', colors='orange')
    ax[1].set_xlim(0, metrics.shape[0])

    losses = ax[1].twinx()
    losses.plot(loss, color='black', linewidth=1)
    losses.tick_params(axis='y')
    losses.set_ylabel("Loss")
    losses.set_xlim(0, metrics.shape[0])
    losses.set_yscale("log")

    return fig


def gif(environment, agent, path="./live-preview.gif", skip=4, duration=25):
    """
    Create a GIF of the agent playing the environment.

    Parameters
    ----------
    environment : gym.Env
    agent : torch.nn.Module
    path : str, optional
        The path to save the GIF.
    skip : int, optional
        The number of frames to skip between observations.
    duration : int, optional
        The duration of each frame in the GIF.
    """
    initial = agent.preprocess(environment.reset()[0])
    states = torch.cat([initial] * agent.shape["reshape"][1], dim=1)

    images = []
    done = False
    while not done:
        _, states, _, done = agent.observe(environment, states, skip)

        images.append(environment.render())
    _ = imageio.mimsave(path, images, duration=duration)
