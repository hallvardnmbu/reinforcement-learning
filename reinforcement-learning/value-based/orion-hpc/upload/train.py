"""
Orion HPC training script.

Value-based vision agent in the tetris environment using PyTorch
"""

import os
import re
import csv
import copy
import time
import logging

import torch
import gymnasium as gym

from agent import VisionDeepQ

# Logging
# --------------------------------------------------------------------------------------------------

handler = logging.FileHandler('./output/info.txt')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Environment
# --------------------------------------------------------------------------------------------------

environment = gym.make('ALE/Tetris-v5', render_mode="rgb_array",
                       obs_type="grayscale", frameskip=4, repeat_action_probability=0.25)
environment.metadata["render_fps"] = 30

# Parameters
# --------------------------------------------------------------------------------------------------
# Description of the parameters:
#   SHAPE : input shape of the network (batch, channels, height, width)
#   DISCOUNT : discount rate for rewards
#   GAMMA : discount rate for Q-learning
#   EXPLORATION_RATE : initial exploration rate
#   EXPLORATION_MIN : minimum exploration rate
#   EXPLORATION_STEPS : number of games to decay exploration rate from `RATE` to `MIN`
#   MINIBATCH : size of the minibatch
#   TRAIN_EVERY : train the network every n games
#   START_TRAINING_AT : start training after n games
#   REMEMBER_ALL : only remember games with rewards
#   MEMORY : size of the agents internal memory
#   RESET_Q_EVERY : update target-network every n games

GAMES = 250000
CHECKPOINT = 2500

SHAPE = (1, 1, 210, 160)
RESHAPE = (1, 1, 203-27, 64-22)  # See method `agent.preprocess` for more information.

DISCOUNT = 0.95
GAMMA = 0.99

PUNISHMENT = -1
INCENTIVE = 1

MINIBATCH = 16
TRAIN_EVERY = 50
START_TRAINING_AT = 5000

EXPLORATION_RATE = 1.0
EXPLORATION_MIN = 0.001
EXPLORATION_STEPS = 2000 // TRAIN_EVERY

REMEMBER_ALL = False
MEMORY = 250
RESET_Q_EVERY = TRAIN_EVERY * 25

NETWORK = {
    "input_channels": 1, "outputs": 5,
    "channels": [32, 32],
    "kernels": [3, 5],
    # "strides": [1, 2],
    "padding": ["same", "same"],
}
OPTIMIZER = {
    "optimizer": torch.optim.AdamW,
    "lr": 0.001,
    "hyperparameters": {}
}

METRICS = "./output/metrics.csv"


# Initialisation
# --------------------------------------------------------------------------------------------------

logger.info("Initialising agent")

value_agent = VisionDeepQ(
    network=NETWORK, optimizer=OPTIMIZER,

    batch_size=MINIBATCH, shape=RESHAPE,

    other={
        "discount": DISCOUNT, "gamma": GAMMA,

        "memory": MEMORY,

        "incentive": INCENTIVE, "punishment": PUNISHMENT,

        "exploration_rate": EXPLORATION_RATE,
        "exploration_steps": EXPLORATION_STEPS,
        "exploration_min": EXPLORATION_MIN
    }
)

# Searching for and loading previous weights
# --------------------------------------------------------------------------------------------------
# Searches for the pattern "weights-{CHECKPOINT}.pth" in the current directory and
# subdirectories, and loads the weights from the file with the highest checkpoint.

files = [os.path.join(root, f)
         for root, dirs, files in os.walk(".")
         for f in files if f.endswith('.pth')]
if files:
    for file in sorted(files,
                       key=lambda x: int(re.search(r'/weights-(\d+).pth', x).group(1))
                       if re.search(r'/weights-(\d+).pth', x) is not None
                       else 0, reverse=True):
        try:
            weights = torch.load(file, map_location=value_agent.device)
            value_agent.load_state_dict(weights)
            logger.info("Weights loaded from %s", file)
            break
        except RuntimeError as e:
            logger.error("Failed to load weights from %s due to error: %s", file, str(e))

# Target-network
# --------------------------------------------------------------------------------------------------

_value_agent = copy.deepcopy(value_agent)

# Metrics
# --------------------------------------------------------------------------------------------------

FILE = open(METRICS, "w", newline="")
METRIC = csv.writer(FILE)
METRIC.writerow(["game", "steps", "loss", "exploration", "reward"])

# Training
# --------------------------------------------------------------------------------------------------

logger.info("Starting playing")
TRAINING = False
_STEPS = _LOSS = _REWARD = 0

start = time.time()
for game in range(1, GAMES + 1):
    if not TRAINING and game >= START_TRAINING_AT:
        logger.info("Starting training")
        TRAINING = True

    state = value_agent.preprocess(environment.reset()[0], SHAPE)

    TERMINATED = TRUNCATED = False

    # LEARNING FROM GAME
    # ----------------------------------------------------------------------------------------------

    STEPS = 0
    REWARDS = 0
    while not (TERMINATED or TRUNCATED):
        action = value_agent.action(state).detach()

        new_state, reward, TERMINATED, TRUNCATED, _ = environment.step(action.item())
        new_state = value_agent.preprocess(new_state, SHAPE)

        value_agent.remember(state, action, torch.tensor([reward]))

        state = new_state

        REWARDS += reward
        STEPS += 1

    if REMEMBER_ALL or REWARDS > 0:
        value_agent.memorize(state, STEPS)
        logger.info("  %s > Rewards: %s Steps: %s Memory: %s",
                    game, int(REWARDS), STEPS, len(value_agent.memory["memory"]))
    value_agent.memory["game"].clear()

    loss = None
    if game % TRAIN_EVERY == 0 and len(value_agent.memory["memory"]) > 0 and TRAINING:
        loss = value_agent.learn(network=_value_agent)
        EXPLORATION_RATE = value_agent.parameter["rate"]
        _LOSS += loss

    if game % RESET_Q_EVERY == 0 and TRAINING:
        logger.info(" Resetting target-network")
        _value_agent.load_state_dict(value_agent.state_dict())

    # METRICS
    # ----------------------------------------------------------------------------------------------

    METRIC.writerow([game, STEPS, loss, EXPLORATION_RATE, REWARDS])
    _REWARD += REWARDS
    _STEPS += STEPS

    if game % CHECKPOINT == 0 or game == GAMES:

        logger.info("Game %s (progress %s %%, random %s %%)",
                    game, int(game * 100 / GAMES), int(EXPLORATION_RATE * 100))
        logger.info(" > Average steps: %s", int(_STEPS / CHECKPOINT))
        logger.info(" > Average loss:  %s", _LOSS / (CHECKPOINT / TRAIN_EVERY))
        logger.info(" > Rewards:       %s", int(_REWARD / CHECKPOINT))
        _STEPS = _LOSS = _REWARD = 0

        if TRAINING:
            logger.info("Saving weights")
            torch.save(value_agent.state_dict(), f"./output/weights-{game}.pth")

FILE.close()
logger.info("Total training time: %s seconds", round(time.time() - start, 2))

torch.save(value_agent.state_dict(), f"./output/weights-{GAMES}.pth")
logger.info(" > Saved final weights to ./output/weights-%s.pth", GAMES)

logger.info("Metrics saved to %s", METRICS)

logger.info("Done")
