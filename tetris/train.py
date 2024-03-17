"""
Orion HPC training script.

Value-based vision agent in the Enduro environment using PyTorch.
"""

import re
import csv
import glob
import copy
import time
import random
import logging

import torch
import gymnasium as gym

from DQN import VisionDeepQ

# Logging
# --------------------------------------------------------------------------------------------------

handler = logging.FileHandler('./output/info.txt')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Environment
# --------------------------------------------------------------------------------------------------

environment = gym.make('ALE/Tetris-v5', render_mode="rgb_array",
                       obs_type="grayscale", frameskip=1, repeat_action_probability=0.0)
environment.metadata["render_fps"] = 30

# Parameters
# --------------------------------------------------------------------------------------------------
# GAMES : The total number of games to be played.
# SKIP : The number of frames to skip between each saved frame.
# CHECKPOINT : The interval at which checkpoints are saved during the training process.
# SHAPE : A dictionary defining the shape of the original image and the slices for height and width.
# DISCOUNT : The discount rate for rewards in the Q-learning algorithm.
# GAMMA : The discount rate for future rewards in the Q-learning algorithm.
# GRADIENTS : The range within which gradients are clamped.
# PUNISHMENT : The punishment value for losing a game.
# INCENTIVE : The incentive value for winning a game.
# MINIBATCH : The size of the minibatch used in training.
# TRAIN_EVERY : The interval at which the network is trained.
# START_TRAINING_AT : The game number at which the training starts.
# EXPLORATION_RATE : The initial exploration rate.
# EXPLORATION_MIN : The minimum exploration rate.
# EXPLORATION_STEPS : The number of games over which the exploration rate decays from RATE to MIN.
# REMEMBER : Remember a game with a certain probability regardless of the reward.
# MEMORY : The size of the agent's internal memory.
# RESET_Q_EVERY : The interval at which the target network is updated.
# NETWORK : A dictionary defining the architecture of the neural network.
# OPTIMIZER : A dictionary defining the optimizer used in training.
# METRICS : The file path where the metrics are saved.

GAMES = 10000
SKIP = 3
CHECKPOINT = 1000

SHAPE = {
    "original": (1, 1, 210, 160),
    "height": slice(27, 203),
    "width": slice(22, 64)
}

DISCOUNT = 0.95
GAMMA = 0.99
GRADIENTS = (-1, 1)

PUNISHMENT = -5
INCENTIVE = 1

MINIBATCH = 64
TRAIN_EVERY = 1
START_TRAINING_AT = 64

EXPLORATION_RATE = 1.0
EXPLORATION_MIN = 0.001
EXPLORATION_STEPS = 5000 // TRAIN_EVERY

REMEMBER = 1.0
MEMORY = 500
RESET_Q_EVERY = TRAIN_EVERY * 5

NETWORK = {
    "input_channels": 4, "outputs": 5,
    "channels": [128, 64],
    "kernels": [2, 2],
    "padding": ["valid", "valid"],
    "strides": [2, 1],
    "nodes": [128],
}
OPTIMIZER = {
    "optimizer": torch.optim.RMSprop,
    "lr": 0.0001,
    "hyperparameters": {}
}

METRICS = "./output/metrics.csv"

# Initialisation
# --------------------------------------------------------------------------------------------------
# Searches for the pattern "weights-{CHECKPOINT}.pth" in the current directory and
# subdirectories, and loads the weights from the file with the highest checkpoint.

logger.debug("Initialising agent")
value_agent = VisionDeepQ(
    network=NETWORK, optimizer=OPTIMIZER, shape=SHAPE,

    batch_size=MINIBATCH, memory=MEMORY,

    discount=DISCOUNT, gamma=GAMMA,
    punishment=PUNISHMENT, incentive=INCENTIVE,

    exploration_rate=EXPLORATION_RATE,
    exploration_steps=EXPLORATION_STEPS,
    exploration_min=EXPLORATION_MIN,
)
logger.info(value_agent.eval())

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

_value_agent = copy.deepcopy(value_agent)

with open(METRICS, "w", newline="", encoding="UTF-8") as file:
    metric = csv.writer(file)
    metric.writerow(["game", "steps", "loss", "exploration", "reward"])

# Training
# --------------------------------------------------------------------------------------------------

logger.info("Started playing")
start = time.time()

TRAINING = False
_STEPS = _LOSS = _REWARD = 0
for game in range(1, GAMES + 1):
    initial = value_agent.preprocess(environment.reset()[0])
    states = torch.cat(
        [initial.view(1, 1, *value_agent.shape["reshape"][2:])] * value_agent.shape["reshape"][1],
        dim=1
    )

    DONE = False
    STEPS = REWARDS = 0
    TRAINING = True if (not TRAINING and game >= START_TRAINING_AT) else TRAINING
    while not DONE:
        action, new_states, rewards, DONE = value_agent.observe(environment, states, SKIP)
        value_agent.remember(states, action, torch.tensor(rewards))

        states = new_states
        REWARDS += rewards
        STEPS += 1

    if random.random() < REMEMBER or REWARDS > 0:
        value_agent.memorize(states, STEPS)
        logger.debug("  %s --> (%s) %s", game, int(STEPS), int(REWARDS))
    value_agent.memory["game"].clear()

    LOSS = None
    if game % TRAIN_EVERY == 0 and TRAINING:
        LOSS = value_agent.learn(network=_value_agent, clamp=GRADIENTS)
        EXPLORATION_RATE = value_agent.parameter["rate"]
        _LOSS += LOSS
    _REWARD += REWARDS
    _STEPS += STEPS

    if game % RESET_Q_EVERY == 0 and TRAINING:
        logger.info(" Resetting target-network")
        _value_agent.load_state_dict(value_agent.state_dict())

    # METRICS
    # ----------------------------------------------------------------------------------------------
    # Saves the metrics to a CSV file. Logs the progress of the training and saves the current
    # weights every `CHECKPOINT` games.

    with open(METRICS, "a", newline="", encoding="UTF-8") as file:
        metric = csv.writer(file)
        metric.writerow([game, STEPS, LOSS, EXPLORATION_RATE, int(REWARDS)])

    if game % (CHECKPOINT // 2) == 0 or game == GAMES:
        logger.info("Game %s (progress %s %%, random %s %%)",
                    game, int(game * 100 / GAMES), round(EXPLORATION_RATE * 100, 2))
        logger.info(" > Average steps: %s", int(_STEPS / (CHECKPOINT // 2)))
        logger.info(" > Average loss:  %s", _LOSS / ((CHECKPOINT // 2) / TRAIN_EVERY))
        logger.info(" > Rewards:       %s", _REWARD)
        _STEPS = _LOSS = _REWARD = 0

    if TRAINING and game % CHECKPOINT == 0:
        logger.info("Saving model")
        torch.save(value_agent.state_dict(), f"./output/weights-{game}.pth")

logger.info("Total training time: %s seconds", round(time.time() - start, 2))
logger.debug("Metrics saved to %s", METRICS)
