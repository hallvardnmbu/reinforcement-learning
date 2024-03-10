"""GIF utilities for visualizing pre-trained agents interacting with environments."""

import torch
import imageio


def gif_single(environment, agent, path="./live-preview.gif", duration=25):
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


def gif_stacked(environment, agent, path="./live-preview.gif", skip=4, duration=25):
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
