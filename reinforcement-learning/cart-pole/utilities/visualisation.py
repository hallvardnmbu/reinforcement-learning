"""Visualisation utilities for the Tetris environment and agents."""

import torch
import imageio
import numpy as np
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


def gif(environment, agent, path="./live-preview.gif", duration=25):
    """
    Create a GIF of the agent playing the environment.

    Parameters
    ----------
    environment : gym.Env
    agent : torch.nn.Module
    path : str, optional
        The path to save the GIF.
    duration : int, optional
        The duration of each frame in the GIF.
    """
    state = torch.tensor(environment.reset()[0], dtype=torch.float32)

    images = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent(state).argmax().item()

        state, _, terminated, truncated, _ = environment.step(action)
        state = torch.tensor(state, dtype=torch.float32)

        images.append(environment.render())
    _ = imageio.mimsave(path, images, duration=duration)
