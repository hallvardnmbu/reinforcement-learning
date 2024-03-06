"""
Value-based agent for reinforcement learning.

Useful for environments with `rgb` or `grayscale` state spaces (from `gymnasium`).

NOTE: MLX convolutional layers has channels as the last dimension, where PyTorch has it as the
second; [B, H, W, C] vs. [B, C, H, W].
"""

from collections import deque, namedtuple
import random

import numpy as np
from mlx import nn
from mlx import core as mx


class Network(nn.Module):
    """Agent network."""
    def __init__(self, network):
        """
        Based vision agent for reinforcement learning.

        Parameters
        ----------
        network : dict
            Contains the architecture for the model.
            The dictionary must contain the following keys:

            input_channels : int
                Number of input channels.
            outputs : int
                Number of output nodes (actions).
            kernels : list of tuple of int
                Kernel size for each layer.
            channels : list, optional
                Number of channels for each hidden layer.
        """
        super().__init__()

        # ARCHITECTURE
        # ------------------------------------------------------------------------------------------
        # The following makes sure the input shapes are correct, and corrects them if not.

        self.layers = []

        if "channels" not in network:
            print("No channels found in input.\n"
                  "Using a default of 15 channels for the convolutional layer.")
            network["channels"] = [15] * network.get("kernels", 1)
        channels = len(network["channels"])

        if "kernels" not in network:
            print(f"No kernels found in input.\n"
                  f"Using a default kernel size 3 for all ({channels}) convolutional layers.")
            network["kernels"] = [3] * channels

        if len(network["kernels"]) != channels:
            print(f"Number of kernels must be equal to the number of layers ({channels}).\n"
                  f"Using default kernel size 3 for all layers.")
            network["kernels"] = [3] * channels

        if len(network["strides"]) != channels:
            print(f"Number of strides must be equal to the number of layers ({channels}).\n"
                  f"Using default strides size 1 for all layers.")
            network["strides"] = [1] * channels

        # Convolutional layers:
        # ------------------------------------------------------------------------------------------

        for _in, _out, _kernel, _stride in (
                zip(
                    [network["input_channels"]] + network["channels"][:-1],
                    network["channels"],
                    network["kernels"],
                    network["strides"]
                )
        ):
            self.layers.append(nn.Conv2d(_in, _out, kernel_size=_kernel, stride=_stride))

        # Calculating the output shape of convolutional layers:
        # ------------------------------------------------------------------------------------------

        _output = mx.zeros([1, 210, 160, network["input_channels"]])
        for layer in self.layers:
            _output = layer(_output)
        _output = _output.flatten().shape[0]

        # Fully connected layers:
        # ------------------------------------------------------------------------------------------

        if "nodes" not in network:
            self.layers.append(nn.Linear(_output, network["outputs"]))
        else:
            self.layers.append(nn.Linear(_output, network["nodes"][0]))

            for _in, _out in (
                    zip(
                        network["nodes"],
                        network["nodes"][1:] + [network["outputs"]])
            ):
                self.layers.append(nn.Linear(_in, _out))

    def __call__(self, state):
        """
        Forward pass with nonmodified output.

        Parameters
        ----------
        state : mlx.core.Array
            Observed state.

        Returns
        -------
        output : mlx.core.Array
        """
        state = state / mx.array(255)
        _output = nn.relu(self.layers[0](state))

        for layer in self.layers[1:-1]:
            if layer.__class__.__name__ == "Linear":
                _output = _output.reshape(_output.shape[0], -1)
            _output = nn.relu(layer(_output))

        _output = _output.reshape(_output.shape[0], -1)
        output = self.layers[-1](_output)

        return output


class VisionDeepQ:
    """Value-based vision agent for reinforcement learning."""
    Memory = namedtuple("Memory",
                        ["state", "action", "new_state", "reward", "steps"])

    def __init__(self,
                 network,
                 optimizer,
                 batch_size=32,
                 **other):
        """
        Value-based vision agent for reinforcement learning.

        Parameters
        ----------
        network : dict
            Contains the architecture for the model.
            The dictionary must contain the following keys:

            input_channels : int
                Number of input channels.
            outputs : int
                Number of output nodes (actions).
            kernels : list of tuple of int
                Kernel size for each layer.
            channels : list, optional
                Number of channels for each hidden layer.
        optimizer : dict
            Contains the optimizer for the model and its hyperparameters. The dictionary must
            contain the following keys:

            optimizer : mlx.optimizers.X
                The optimizer for the model.
            lr : float
                Learning rate for the optimizer.
            **hyperparameters : dict, optional
                Additional hyperparameters for the optimizer.
        other
            Additional parameters.

            discount : float, optional
                Discount factor for future rewards.
                --> 0: only consider immediate rewards
                --> 1: consider all future rewards equally
            gamma : float, optional
                Discount factor for Q-learning.
        """
        self.agent = Network(network)

        # LEARNING
        # ------------------------------------------------------------------------------------------
        # Default discount factor is 0.99, as suggested by the Google DeepMind paper "Human-level
        # control through deep reinforcement learning" (2015).

        self.gradient = nn.value_and_grad(self.agent, self.loss)

        self.parameter = {
            "rate": other.get("exploration_rate", 0.9),
            "decay": other.get("exploration_decay", 0.995),
            "min": other.get("exploration_min", 0.01),

            "discount": other.get("discount", 0.99),
            "gamma": other.get("gamma", 0.95),
        }

        self.optimizer = optimizer["optimizer"](learning_rate=optimizer["lr"],
                                                **optimizer.get("hyperparameters", {}))

        self.memory = {
            "batch_size": batch_size,
            "memory": deque(maxlen=other.get("memory", 2500)),
            "game": []
        }

    def action(self, state):
        """
        Greedy action selection with stochastic exploration.

        Parameters
        ----------
        state : mlx.core.Array
            Observed state.

        Returns
        -------
        action : int
            Selected action.
        actions : mlx.core.Array
            Q-values for each action.
        """
        if np.random.rand() < self.parameter["rate"]:
            action = mx.array(np.random.choice(range(self.agent.layers[-1].weight.shape[0])))
        else:
            action = mx.argmax(self.agent(state))

        return action

    def learn(self, network):
        """
        Q-learning algorithm; a value-based method.

        Parameters
        ----------
        network : mlx.nn.Module
            Reference network for Q-learning.

        Returns
        -------
        loss : float
            Relative loss.

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

        states = mx.concatenate([mx.stack(game.state).squeeze()
                                 for game in memory]).reshape(-1, 210, 160, 1)
        actions = mx.concatenate([mx.stack(game.action)
                                  for game in memory]).reshape(-1, 1)
        new_states = mx.concatenate([mx.stack(game.new_state).squeeze()
                                     for game in memory]).reshape(-1, 210, 160, 1)
        rewards = mx.concatenate([mx.stack(game.reward)
                                  for game in memory])

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
            _reward = _reward * self.parameter["discount"] + rewards[i]
            rewards[i] = _reward

        mean = mx.mean(rewards)
        std = mx.sqrt(mx.sum((rewards - mean) ** 2) / rewards.shape[0])
        rewards = (rewards - mean) / (std + 1e-7)

        # GRADIENT
        # ------------------------------------------------------------------------------------------
        # See the `loss` method for more information.

        loss, gradients = self.gradient(states, actions, new_states, rewards, steps, network)

        # BACKPROPAGATION
        # ------------------------------------------------------------------------------------------

        self.optimizer.update(self.agent, gradients)

        # EXPLORATION RATE DECAY
        # ------------------------------------------------------------------------------------------

        self.parameter["rate"] = max(self.parameter["decay"] * self.parameter["rate"],
                                     self.parameter["min"])

        return loss.item() * (1000 / steps[-1])

    def loss(self, *data):
        r"""
        Compute the policy gradient through Q-learning.

        Parameters
        ----------
        data : list
            Must contain the following items in this order:

            states : mlx.core.array
                Observed states.
            actions : mlx.core.array
                Selected actions.
            new_states : mlx.core.array
                Observed states after action.
            rewards : mlx.core.array
                Expected rewards.
            steps : list
                Number of steps in each game.
            network : nn.Module
                Reference network for Q-learning.

        Returns
        -------
        loss : mlx.core.array

        Notes
        -----
            The Q-learning implementation is based on the letter by Google DeepMind ("Human-level
            control through deep reinforcement learning", 2015). Which is based on the Bellman
            equation for the optimal action-value function. The optimal action-value function is

        .. math::
                Q*(s, a) = Q(s, a) + \alpha * (r + \gamma * \max_a' Q'(s', a') - Q(s, a))

            where :math:`Q*(s, a)` is the optimal action-value function, :math:`Q(s, a)` is the
            current action-value, :math:`Q'(s', a')` is the approximated action-value,
            :math:`\alpha` is the learning rate, :math:`r` is the expected reward and
            :math:`\gamma` is the discount factor.

            DeepMind further simplifies this equation to:

        .. math::
                Q*(s, a) = r + gamma * max_a' Q'(s', a')

            where :math:`Q'` is a copy of the agent, which is updated every :math:`C` steps.
        """
        states, actions, new_states, rewards, steps, network = data

        # Here, the documentation for MLX was useful:
        actual = mx.take_along_axis(self.agent(states),                                 # noqa
                                    indices=actions, axis=1).squeeze()

        optimal = (rewards.reshape(-1) +                                                      # noqa
                   self.parameter["gamma"] * network(new_states).max(axis=1))                   #
        # noqa

        # As Google DeepMind suggests, the optimal Q-value is set to r if the game is over.
        for step in steps:
            optimal[step] = rewards[step]

        loss = nn.losses.mse_loss(actual, optimal)

        return loss

    def remember(self, *args):
        """
        Append state, action, new_state and reward to agents memory of the current game.

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
