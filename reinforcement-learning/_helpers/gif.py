import torch
import imageio


def gif(
        environment,
        agent,
        path="./output/live-preview.gif"
):
    state = torch.tensor(environment.reset()[0], dtype=torch.float32)

    images = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent(state).argmax().item()

        state, reward, terminated, truncated, _ = environment.step(action)
        state = torch.tensor(state, dtype=torch.float32)

        images.append(environment.render())
    _ = imageio.mimsave(path, images, duration=25)


def gif_stack(
        environment,
        agent,
        shape=(1, 1, 210, 160),
        reshape=(1, 4, 88, 21),
        path="./output/live-preview.gif"
):
    initial = agent.preprocess(environment.reset()[0], shape)
    states = torch.cat([initial] * reshape[1], dim=1)

    images = []
    done = False
    while not done:
        _, new_states, _, done = agent.observe(environment, states, shape)

        images.append(environment.render())
    _ = imageio.mimsave(path, images, duration=25)
