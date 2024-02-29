"""Orion HPC training script."""


# Value-based vision agent in the tetris environment using PyTorch
# --------------------------------------------------------------------------------------------------

import copy
import time
import logging

import torch
import imageio
import gymnasium as gym
import matplotlib.pyplot as plt

from agent import VisionDeepQ


handler = logging.FileHandler('./output/debug.txt')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

environment = gym.make('ALE/Tetris-v5', render_mode="rgb_array",
                       obs_type="grayscale", frameskip=4, repeat_action_probability=0.25)
environment.metadata["render_fps"] = 30

# Training
# --------------------------------------------------------------------------------------------------

# Parameters

GAMES = 5000
FRAMESKIP = 4  # Repeat action for n frames

DISCOUNT = 0.99  # Discount rate for rewards
GAMMA = 0.95  # Discount rate for Q-learning

EXPLORATION_RATE = 1.0  # Initial exploration rate
EXPLORATION_DECAY = 0.9995  # Decay rate every game (rate *= decay)
EXPLORATION_MIN = 0.01  # Minimum exploration rate

MINIBATCH = 32  # Size of the minibatch
TRAIN_EVERY = 10  # Train the network every n games
START_TRAINING_AT = 250  # Start training after n games

REMEMBER_ALL = False  # Only remember games with rewards
MEMORY = 1500  # Size of the agents internal memory
RESET_Q_EVERY = 250  # Update target-network every n games

NETWORK = {
    "input_channels": 1, "outputs": 5,
    "frames": FRAMESKIP,
    "channels": [32, 64, 64],
    "kernels": [8, 4, 3],
    "strides": [4, 2, 1],
    "nodes": [64]
}
OPTIMIZER = {
    "optimizer": torch.optim.RMSprop,
    "lr": 0.0025,
    "hyperparameters": {}
}

# Initialisation


logger.debug("Initialising agent")

value_agent = VisionDeepQ(
    network=NETWORK, optimizer=OPTIMIZER,

    discount=DISCOUNT, gamma=GAMMA,

    batch_size=MINIBATCH, memory=MEMORY,

    exploration_rate=EXPLORATION_RATE, exploration_decay=EXPLORATION_DECAY,
    exploration_min=EXPLORATION_MIN
)

_value_agent = copy.deepcopy(value_agent)

CHECKPOINT = GAMES // 10
METRICS = {
    "steps": torch.zeros(GAMES),
    "losses": torch.zeros(GAMES // TRAIN_EVERY),
    "exploration": torch.zeros(GAMES),
    "rewards": torch.zeros(GAMES)
}


# Training
# --------------------------------------------------------------------------------------------------


start = time.time()
for game in range(1, GAMES + 1):

    logger.info("Game %s", game)

    state = torch.tensor(environment.reset()[0], dtype=torch.float32).view((1, 1, 210, 160))  # noqa
    TERMINATED = TRUNCATED = False

    # LEARNING FROM GAME
    # ----------------------------------------------------------------------------------------------

    STEPS = 0
    REWARDS = 0
    while not (TERMINATED or TRUNCATED):
        action = value_agent.action(state)

        logger.debug(" > Action: %s", action.item())

        new_state, reward, TERMINATED, TRUNCATED, _ = environment.step(action.item())  # noqa

        logger.debug("   New state shape before reshaping: %s", new_state.shape)

        new_state = torch.tensor(new_state, dtype=torch.float32).view((1, 1, 210, 160))

        value_agent.remember(state, action, new_state, torch.tensor([reward]))

        logger.debug("   Remembered state (%s), action (%s), new_state (%s) and reward (%s)",
                     state.shape, action.item(), new_state.shape, reward)

        state = new_state

        STEPS += 1
        REWARDS += reward

        logger.info(" > Step %s", STEPS)
        logger.debug(" > Reward: %s", reward)

    if REMEMBER_ALL or REWARDS > 0:
        logger.debug(" > Memorizing game")
        value_agent.memorize(STEPS)
        logger.debug(" > Memorized. Memory size: %s", len(value_agent.memory["game"]))
    else:
        logger.debug(" > Not memorizing game")
        value_agent.memory["game"].clear()

    if (game % TRAIN_EVERY == 0
            and len(value_agent.memory["memory"]) > 0
            and game >= START_TRAINING_AT):

        logger.info(" > Training agent")

        loss = value_agent.learn(network=_value_agent)
        METRICS["losses"][game // TRAIN_EVERY - 1] = loss

        logger.info("   Loss: %s", loss)

    if game % RESET_Q_EVERY == 0 and game > START_TRAINING_AT:
        logger.info(" > Resetting target-network")

        _value_agent.load_state_dict(value_agent.state_dict())

        logger.info("   Target-network reset")

    # METRICS
    # ----------------------------------------------------------------------------------------------

    METRICS["steps"][game - 1] = STEPS
    METRICS["exploration"][game - 1] = value_agent.parameter["rate"]
    METRICS["rewards"][game - 1] = REWARDS

    if game % CHECKPOINT == 0 or game == GAMES:

        logger.info(" > Saving weights")

        torch.save(value_agent.state_dict(), f"./output/weights-{game}.pth")

        logger.info("   Weights saved to ./output/weights-%s.pth", game)

        _MEAN_STEPS = METRICS["steps"][max(0, game - CHECKPOINT - 1):game - 1].mean()
        _TOTAL_REWARDS = METRICS["rewards"][max(0, game - CHECKPOINT - 1):game - 1].sum()

        if game >= START_TRAINING_AT:
            _MEAN_LOSS = METRICS["losses"][max(0, (game - CHECKPOINT - 1)
                                               // TRAIN_EVERY):game // TRAIN_EVERY].mean()
            _MEAN_LOSS = f"{_MEAN_LOSS:.4f}"
        else:
            _MEAN_LOSS = "-"

        print(f"Game {game:>6} {int(game / GAMES * 100):>16} % \n"
              f"{'-' * 30} \n"
              f" > Average steps: {int(_MEAN_STEPS):>12} \n"
              f" > Average loss: {_MEAN_LOSS:>13} \n"
              f" > Rewards: {int(_TOTAL_REWARDS):>18} \n ")

print(f"Total training time: {time.time() - start:.2f} seconds")

logger.info("Saving final weights")
torch.save(value_agent.state_dict(), "./output/weights-final.pth")
logger.info("   Saved final weights to ./output/weights-final.pth")

# Visualisation
# --------------------------------------------------------------------------------------------------

# Metrics

logger.info("Visualising METRICS")


def moving_average(data, window_size=50):
    """Compute moving average with given window size of the data."""
    half_window = window_size // 2
    return [(data[max(0, i - half_window):min(len(data), i + half_window)]).mean()
            for i in range(len(data))]


STEPS = moving_average(METRICS["steps"])
LOSSES = moving_average(METRICS["losses"])
REWARDS = [val.item() if val > 0 else torch.nan for val in METRICS["rewards"]]

fig, ax = plt.subplots(3, 1, figsize=(12, 8))
fig.suptitle("Value-based: vision deep Q-learning agent")

ax[0].plot(STEPS, color="black", linewidth=1)
ax[0].set_xticks([])
ax[0].set_title("Average steps per game")

ax[1].plot(torch.linspace(0, GAMES, len(LOSSES)), LOSSES, color="black", linewidth=1)
ax[1].set_yscale("log") if any(loss > 0 for loss in LOSSES) else None
ax[1].set_xticks([])
ax[1].set_title("Average loss")

ax_2 = ax[1].twinx()
ax_2.plot(METRICS["exploration"], color="gray", linewidth=0.5)
ax_2.set_ylabel("Exploration rate")
ax_2.yaxis.label.set_color('gray')
ax_2.set_ylim(-0.1, 1.1)
ax_2.tick_params(axis='y', colors='gray')

ax[2].scatter(range(len(REWARDS)), REWARDS, color="black", s=15, marker="*")
ticks = list(set(reward for reward in REWARDS if not torch.isnan(torch.tensor(reward))))
ax[2].set_yticks(ticks) if ticks else None
ax[2].set_xlim(ax[1].get_xlim())
ax[2].set_xlabel("Game nr.")
ax[2].set_title("Rewards per game")

for i in range(0, GAMES, GAMES // 10):
    ax[0].axvline(x=i, color='gray', linewidth=0.5)
    ax[1].axvline(x=i, color='gray', linewidth=0.5)
    ax[2].axvline(x=i, color='gray', linewidth=0.5)

ax[0].axvline(x=START_TRAINING_AT, color='black', linewidth=1)
ax[1].axvline(x=START_TRAINING_AT, color='black', linewidth=1)
ax[2].axvline(x=START_TRAINING_AT, color='black', linewidth=1)
ax[2].text(START_TRAINING_AT, 1, 'Training starts',
           rotation=90, verticalalignment='center', horizontalalignment='center',
           bbox={"facecolor": 'white', "alpha": 1.0})

plt.savefig("./output/value-vision-tetris.png")

# In action

logger.info("Running agent in environment")

state = torch.tensor(environment.reset()[0], dtype=torch.float32).view((1, 1, 210, 160))

images = []
TERMINATED = TRUNCATED = False
while not (TERMINATED or TRUNCATED):
    action = value_agent(state).argmax(1).item()

    state, reward, TERMINATED, TRUNCATED, _ = environment.step(action)
    state = torch.tensor(state, dtype=torch.float32).view((1, 1, 210, 160))

    images.append(environment.render())
_ = imageio.mimsave('./output/value-vision-tetris.gif', images, duration=50)

logger.info("Done")
