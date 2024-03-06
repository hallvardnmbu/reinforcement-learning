"""
Orion HPC training script.

Value-based vision agent in the tetris environment using PyTorch
"""

import re
import csv
import glob
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
#   PUNISHMENT : punishment for losing
#   INCENTIVE : incentive for rewards
#   EXPLORATION_RATE : initial exploration rate
#   EXPLORATION_MIN : minimum exploration rate
#   EXPLORATION_STEPS : number of games to decay exploration rate from `RATE` to `MIN`
#   MINIBATCH : size of the minibatch
#   TRAIN_EVERY : train the network every n games
#   START_TRAINING_AT : start training after n games
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

    memory=MEMORY,

    discount=DISCOUNT, gamma=GAMMA,

    punishment=PUNISHMENT, incentive=INCENTIVE,

    exploration_rate=EXPLORATION_RATE,
    exploration_steps=EXPLORATION_STEPS,
    exploration_min=EXPLORATION_MIN,
)

with open(METRICS, "w", newline="") as file:
    metric = csv.writer(file)
    metric.writerow(["game", "steps", "loss", "exploration", "reward"])

# Searching for and loading previous weights
# --------------------------------------------------------------------------------------------------
# Searches for the pattern "weights-{CHECKPOINT}.pth" in the current directory and
# subdirectories, and loads the weights from the file with the highest checkpoint.

files = glob.glob("**/*.pth", recursive=True)
if files:
    for file in sorted(files, key=lambda x: int(re.search(r'/weights-(\d+).pth', x).group(1))
                       if re.search(r'/weights-(\d+).pth', x)
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

# Training
# --------------------------------------------------------------------------------------------------

logger.info("Starting playing")
start = time.time()

TRAINING = False
_STEPS = _LOSS = _REWARD = 0
for game in range(1, GAMES + 1):
    state = value_agent.preprocess(environment.reset()[0], SHAPE)

    STEPS = REWARDS = 0
    TERMINATED = TRUNCATED = False
    TRAINING = True if (not TRAINING and game >= START_TRAINING_AT) else TRAINING
    while not (TERMINATED or TRUNCATED):
        action = value_agent.action(state).detach()

        new_state, reward, TERMINATED, TRUNCATED, _ = environment.step(action.item())
        new_state = value_agent.preprocess(new_state, SHAPE)

        value_agent.remember(state, action, torch.tensor([reward]))

        state = new_state
        REWARDS += reward
        STEPS += 1

    if REWARDS > 0:
        value_agent.memorize(state, STEPS)
        logger.info("  %s > Rewards: %s Steps: %s Memory: %s",
                    game, int(REWARDS), STEPS, len(value_agent.memory["memory"]))
    value_agent.memory["game"].clear()

    loss = None
    if game % TRAIN_EVERY == 0 and len(value_agent.memory["memory"]) > 0 and TRAINING:
        loss = value_agent.learn(network=_value_agent)
        EXPLORATION_RATE = value_agent.parameter["rate"]
        _LOSS += loss
    _REWARD += REWARDS
    _STEPS += STEPS

    if game % RESET_Q_EVERY == 0 and TRAINING:
        logger.info(" Resetting target-network")
        _value_agent.load_state_dict(value_agent.state_dict())

    # METRICS
    # ----------------------------------------------------------------------------------------------
    # Saves the metrics to a CSV file. Logs the progress of the training and saves the current
    # weights every `CHECKPOINT` games.

    with open(METRICS, "a", newline="") as file:
        metric = csv.writer(file)
        metric.writerow([game, STEPS, loss, EXPLORATION_RATE, REWARDS])

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

logger.info("Total training time: %s seconds", round(time.time() - start, 2))
logger.info("Metrics saved to %s", METRICS)
