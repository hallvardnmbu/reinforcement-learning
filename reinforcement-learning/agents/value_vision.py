"""
Value-based agent for reinforcement learning.

Useful for environments with `rgb` or `grayscale` state spaces (from `gymnasium`). See
`value_simple.py` or `value_advanced.py` for other implementations.
"""

from collections import deque, namedtuple
import random

import numpy as np
import torch


class VisionDeepQ(torch.nn.Module):
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

            optimizer : torch.optim.X
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
            gamma : float, optional
                Discount factor for Q-learning.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ARCHITECTURE
        # ------------------------------------------------------------------------------------------
        # The following makes sure the input shapes are correct, and corrects them if not.

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

        for i, (_in, _out, _kernel, _stride) in (
                enumerate(zip(
                    [network["input_channels"]] + network["channels"][:-1],
                    network["channels"],
                    network["kernels"],
                    network["strides"]
                ))
        ):
            setattr(self, f"layer_{i}",
                    torch.nn.Conv2d(_in, _out, kernel_size=_kernel, stride=_stride))

        self.convolutions = len(network["channels"]) - len(network.get("nodes", []))

        # Calculating the output shape of convolutional layers:
        # ------------------------------------------------------------------------------------------

        with torch.no_grad():
            _output = torch.zeros([1, network["input_channels"], 210, 160])
            for layer in self._modules.values():
                _output = layer(_output)
            _output = _output.view(_output.size(0), -1).flatten().shape[0]

        # Fully connected layers:
        # ------------------------------------------------------------------------------------------

        if "nodes" not in network:
            setattr(self, f"layer_{len(network['channels'])}",
                    torch.nn.Linear(_output, network["outputs"], dtype=torch.float32))
        else:
            setattr(self, f"layer_{len(network['channels'])}",
                    torch.nn.Linear(_output, network["nodes"][0], dtype=torch.float32))

            for i, (_in, _out) in (
                    enumerate(zip(
                        network["nodes"],
                        network["nodes"][1:] + [network["outputs"]]))
            ):
                setattr(self, f"layer_{len(network['channels'])+i+1}",
                        torch.nn.Linear(_in, _out, dtype=torch.float32))

        # LEARNING
        # ------------------------------------------------------------------------------------------
        # Default discount factor is 0.99, as suggested by the Google DeepMind paper "Human-level
        # control through deep reinforcement learning" (2015).

        self.parameter = {
            "rate": other.get("exploration_rate", 0.9),
            "decay": other.get("exploration_decay", 0.995),
            "min": other.get("exploration_min", 0.01),

            "discount": other.get("discount", 0.99),
            "gamma": other.get("gamma", 0.95),
        }

        self.optimizer = optimizer["optimizer"](self.parameters(), lr=optimizer["lr"],
                                                **optimizer.get("hyperparameters", {}))

        self.batch_size = batch_size
        self.memory = deque(maxlen=other.get("memory", 2500))
        self.game = []

        self.to(self.device)

    def forward(self, state):
        """
        Forward pass with nonmodified output.

        Parameters
        ----------
        state : torch.Tensor
            Observed state.

        Returns
        -------
        output : torch.Tensor
        """
        state = state.to(self.device) / torch.tensor(255,
                                                     dtype=torch.float32, device=self.device)

        print(state.shape)

        _output = torch.relu(self.layer_0(state))

        for i in range(1, len(self._modules) - 1):
            if i > self.convolutions:
                _output = _output.view(_output.size(0), -1)

            print(_output.shape)

            _output = torch.relu(getattr(self, f"layer_{i}")(_output))

        _output = _output.view(_output.size(0), -1)

        output = getattr(self, f"layer_{len(self._modules)-1}")(_output)

        return output

    def action(self, state):
        """
        Greedy action selection with stochastic exploration.

        Parameters
        ----------
        state : torch.Tensor
            Observed state.

        Returns
        -------
        action : int
            Selected action.
        actions : torch.Tensor
            Q-values for each action.
        """
        if np.random.rand() < self.parameter["rate"]:
            action = torch.tensor([np.random.choice(
                next(reversed(self._modules.values())).out_features
            )], dtype=torch.long)
        else:
            action = self(state).max(1).indices.flatten()

        return action

    def learn(self, network):
        """
        Q-learning algorithm; a value-based method.

        Parameters
        ----------
        network : torch.nn.Module
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
        memory = random.sample(self.memory, min(self.batch_size, len(self.memory)))

        states = torch.cat([torch.stack(game.state).squeeze() for game in memory]).unsqueeze(1)
        actions = torch.cat([torch.stack(game.action) for game in memory]).to(self.device)
        new_states = torch.cat([torch.stack(game.new_state).squeeze()
                                for game in memory]).unsqueeze(1)
        rewards = torch.cat([torch.stack(game.reward) for game in memory]).to(self.device)

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
        rewards = ((rewards - rewards.mean()) / (rewards.std() + 1e-9)).view(-1, 1)

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

        actual = self(states).gather(1, actions.view(-1, 1))

        with torch.no_grad():
            optimal = (rewards +
                       self.parameter["gamma"] * network(new_states).max(1).values.view(-1, 1))

        # As Google DeepMind suggests, the optimal Q-value is set to r if the game is over.
        for step in steps:
            optimal[step] = rewards[step]

        # BACKPROPAGATION
        # ------------------------------------------------------------------------------------------

        loss = torch.nn.functional.smooth_l1_loss(actual, optimal)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # EXPLORATION RATE DECAY
        # ------------------------------------------------------------------------------------------

        self.parameter["rate"] = max(self.parameter["decay"] * self.parameter["rate"],
                                     self.parameter["min"])

        del states, actions, new_states, rewards, _reward, actual, optimal

        return (loss.item() / steps[-1]) * 10000

    def remember(self, *args):
        """
        Append state, action, new_state and reward to agents memory of the current game.

        Parameters
        ----------
        *args : list
            Positional arguments to memorize.
        """
        self.game.append(args)

    def memorize(self, steps):
        """
        Append game to agent memory for mini-batch training.

        Parameters
        ----------
        steps : int
            Number of steps in the game (i.e., game length).
        """
        self.memory.append(self.Memory(*zip(*self.game), steps))
        self.game = []
