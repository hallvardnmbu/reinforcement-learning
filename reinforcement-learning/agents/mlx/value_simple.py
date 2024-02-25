"""Value-based agents."""

from collections import deque, namedtuple
import random

import numpy as np
from mlx import nn
from mlx import core as mx


class MinibatchDeepQ(nn.Module):
    """Value-based agent for reinforcement learning."""
    Memory = namedtuple("Memory",
                        ["state", "new_state", "reward", "steps"])

    def __init__(self,
                 network,
                 optimizer,
                 batch_size=32,
                 **other):
        """
        Base agent for reinforcement learning.

        Parameters
        ----------
        network : dict
            Contains the architecture for the model.
            The dictionary must contain the following keys:

            inputs : int
                Number of input nodes (observations).
            outputs : int
                Number of output nodes (actions).
            nodes : list, optional
                Number of nodes for each hidden layer.
        optimizer : dict
            Contains the optimizer for the model and its hyperparameters. The dictionary must
            contain the following keys:

            optim : mlx.optimizers.X
                The optimizer for the model.
            lr : float
                Learning rate for the optimizer.
            **hyperparameters : dict, optional
                Additional hyperparameters for the optimizer.
        other : dict
            Additional parameters.

            discount : float, optional
                Discount factor for future rewards.
                --> 0: only consider immediate rewards
                --> 1: consider all future rewards equally
        """
        super().__init__()

        # ARCHITECTURE
        # ------------------------------------------------------------------------------------------

        if "nodes" not in network:
            network["nodes"] = [25]

        self.layers = []
        for (_in, _out) in zip([network["inputs"]] + network["nodes"],
                               network["nodes"] + [network["outputs"]]):
            self.layers.append(nn.Linear(_in, _out))

        # LEARNING
        # ------------------------------------------------------------------------------------------
        # Default discount factor is 0.99, as suggested by the Google DeepMind paper "Human-level
        # control through deep reinforcement learning" (2015).

        self.discount = other.get("discount", 0.99)
        self.gamma = other.get("gamma", 0.95)

        self.explore = {
            "rate": other.get("exploration_rate", 0.9),
            "decay": other.get("exploration_decay", 0.999),
            "min": other.get("exploration_min", 0.01),
        }

        self.optimizer = optimizer["optim"](learning_rate=optimizer["lr"],
                                            **optimizer.get("hyperparameters", {}))

        self.memory = {
            "batch_size": batch_size,
            "memory": deque(maxlen=other.get("memory", 2500)),
            "game": [],
        }

    def __call__(self, state):
        """
        Forward pass with nonmodified output.

        Parameters
        ----------
        state : mlx.core.array
            Observed state.

        Returns
        -------
        output : mlx.core.array
        """
        for layer in self.layers[:-1]:
            state = nn.relu(layer(state))
        output = self.layers[-1](state)

        return output

    def action(self, state):
        """
        Greedy action selection with stochastic exploration.

        Parameters
        ----------
        state : mlx.core.array
            Observed state.

        Returns
        -------
        action : int
            Selected action.
        """
        if np.random.rand() < self.explore["rate"]:
            action = mx.array(np.random.choice(range(self.layers[-1].weight.shape[0])))
        else:
            action = self(state).argmax(axis=0)

        return action

    def learn(self, network):
        """
        Q-learning algorithm; a value-based method.

        Parameters
        ----------
        network : nn.Module
            Reference network for Q-learning.

        Returns
        -------
        loss : float

        Raises
        ------
        ValueError
            If no reference network is passed.

        Notes
        -----
        In order for the agent to best learn the optimal actions, it is common to evaluate the
        expected future rewards. Then, the agent can adjust its predicted action values so that
        this expected reward is maximized.
        """
        memory = random.sample(self.memory["memory"],
                               min(self.memory["batch_size"], len(self.memory["memory"])))

        states = mx.concatenate([mx.array(mx.array(game.state)) for game in memory])
        new_states = mx.concatenate([mx.array(game.new_state) for game in memory])
        rewards = mx.concatenate([mx.array(game.reward) for game in memory])

        steps = [game.steps for game in memory]
        steps = [sum(steps[:i+1])-1 for i in range(len(steps))]

        # EXPECTED FUTURE REWARDS
        # ------------------------------------------------------------------------------------------
        # The expected reward given an action is the sum of all future (discounted) rewards. This is
        # achieved by reversely adding the observed reward and the discounted cumulative future
        # rewards. The rewards are then standardized.

        _reward = 0
        for i in reversed(range(len(rewards))):
            _reward = 0 if i in steps else _reward
            _reward = _reward * self.discount + rewards[i]
            rewards[i] = _reward

        mean = rewards.mean(axis=0)
        std = mx.sqrt(mx.sum(rewards - mean) / len(rewards))
        rewards = (rewards - mean) / (std + 1e-9)

        # Q-LEARNING
        # ------------------------------------------------------------------------------------------
        # The Q-learning implementation is based on the letter by Google DeepMind ("Human-level
        # control through deep reinforcement learning", 2015). Which is based on the Bellman
        # equation for the optimal action-value function. The optimal action-value function is
        #
        #  Q*(s, a) = Q(s, a) + alpha * (r + gamma * max_a' Q'(s', a') - Q(s, a))
        #
        # where Q*(s, a) is the optimal action-value function, Q(s, a) is the current
        # action-value, Q'(s', a') is the approximated action-value, alpha is the learning rate,
        # r is the expected reward and gamma is the discount factor.
        #
        # DeepMind further simplifies this equation to:
        #
        #  Q*(s, a) = r + gamma * max_a' Q'(s', a')
        #
        # where Q' is a copy of the agent, which is updated every C steps.

        actual = self(states).max(axis=1)

        optimal = rewards.reshape(-1) + self.gamma * network(new_states).max(axis=1)

        # As Google DeepMind suggests, the optimal Q-value is set to r if the game is over.
        for step in steps:
            optimal[step] = rewards[step]

        # BACKPROPAGATION
        # ------------------------------------------------------------------------------------------

        loss = nn.losses.mse_loss(actual, optimal)

        self.optimizer.update(self, loss)

        # EXPLORATION RATE DECAY
        # ------------------------------------------------------------------------------------------

        self.explore["rate"] = max(self.explore["decay"] * self.explore["rate"],
                                   self.explore["min"])

        return loss

    def remember(self, *args):
        """
        Append state, new_state and reward to agents memory of the current game.

        Parameters
        ----------
        *args : list
            Positional arguments to memorize.
        """
        self.memory["game"].append(args)

    def memorize(self, steps):
        """
        Append game to agent memory for mini-batch training.

        Parameters
        ----------
        steps : int
            Number of steps in the game (i.e., game length).
        """
        self.memory["memory"].append(self.Memory(*zip(*self.memory["game"]), steps))
        self.memory["game"] = []
