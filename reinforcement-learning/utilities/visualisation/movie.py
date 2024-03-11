"""Create a movie of an agent interacting with an environment."""

import cv2
import torch


def movie(environment, agent, path="./live-preview.gif", skip=4, fps=50):
    """Created by Mistral Large."""
    states = agent.preprocess(environment.reset()[0])
    if hasattr(agent, "shape") and "reshape" in agent.shape:
        states = torch.cat([states] * agent.shape["reshape"][1], dim=1)

    done = False
    height, width, _ = environment.render().shape
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))   # noqa

    while not done:
        _, states, _, done = agent.observe(environment, states, skip)
        writer.write(environment.render())

    writer.release()
