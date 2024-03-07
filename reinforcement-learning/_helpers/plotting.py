import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _roll(data, window=50):
    half_window = window // 2
    return [(data[max(0, i - half_window):min(len(data), i + half_window)]).mean()
            for i in range(len(data))]


def visualise_dict(metrics: dict, title: str, window_size: int = 50) -> plt.Figure:
    steps = _roll(metrics["steps"], window_size)
    losses = _roll(metrics["losses"], window_size)

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


def visualise_csv(path: str, title: str, window: int = 150) -> plt.Figure:
    metrics = pd.read_csv(path, header=0).set_index("game", drop=True)

    steps = metrics["steps"].rolling(window=window, center=True).mean()
    loss = metrics["loss"].dropna().rolling(window=window, min_periods=1).mean()
    rewards = metrics.loc[metrics["reward"] != 0, "reward"]

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(title + f" (window size {window})")

    ax[0].plot(steps, color='black', linewidth=1)
    ax[0].set_xlim(0, metrics.shape[0])
    ax[0].set_ylabel("Average steps per game")

    ax[1].scatter(rewards.index, rewards.values, marker='*', color='orange', s=25)
    ax[1].set_yticks(list(set(metrics["reward"].unique()) - {0.0}))
    ax[1].set_ylabel("\u2605 Reward")
    ax[1].set_xlim(0, metrics.shape[0])

    axs = ax[1].twinx()
    axs.plot(loss, color='gray', linewidth=1)
    axs.set_xlim(0, metrics.shape[0])
    axs.set_ylabel("Loss")
    axs.set_yscale("log")

    return fig
