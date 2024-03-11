"""Create a movie of an agent interacting with an environment."""

import cv2
import torch


def create_movie(environment, agent, path="./live-preview.gif", skip=4, fps=50):
    """Created by Mistral Large."""
    initial = agent.preprocess(environment.reset()[0])
    try:
        states = torch.cat([initial] * agent.shape["reshape"][1], dim=1)
    except AttributeError:
        states = initial

    done = False

    height, width, _ = environment.render().shape
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # noqa
    movie = cv2.VideoWriter(path, fourcc, fps, (width, height))

    while not done:
        _, states, _, done = agent.observe(environment, states, skip)
        movie.write(environment.render())

    cv2.destroyAllWindows()
