"""
Value-based agent for reinforcement learning.

Useful for environments with `rgb` or `grayscale` state spaces (from `gymnasium`) with
frame-stacking (video).
"""

from collections import deque, namedtuple
import random

import numpy as np
import torch


class VideoDeepQ(torch.nn.Module):
    """Value-based vision agent for reinforcement learning."""
    Memory = namedtuple("Memory",
                        ["state", "action", "new_states", "reward", "steps"])
    Game = namedtuple("Game",
                      ["state", "action", "new_states", "reward"])

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
            frames : int
                Number of frames to observe between each action.
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
        other
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

        # Conv3d expects input in the shape of (batch, channels, frames, height, width).
        self.shape = (1, network["input_channels"], network["frames"], 210, 160)

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
                    torch.nn.Conv3d(_in, _out, kernel_size=_kernel, stride=_stride))

        self.convolutions = len(network["channels"]) - len(network.get("nodes", []))

        # Calculating the output shape of convolutional layers:
        # ------------------------------------------------------------------------------------------

        with torch.no_grad():
            _output = torch.zeros(self.shape)
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

        self.memory = {
            "batch_size": batch_size,
            "memory": deque(maxlen=other.get("memory", 2500)),
            "game": deque([]),
            "observed": deque(maxlen=network["frames"]),
        }

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

        _output = torch.relu(self.layer_0(state))

        for i in range(1, len(self._modules) - 1):
            if i > self.convolutions:
                _output = _output.view(_output.size(0), -1)

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
            action = self(state).argmax(1)

        return action

    def observe(self, environment, states):
        """
        Observe the environment for n frames.

        Parameters
        ----------
        environment : gymnasium.Env
            The environment to observe.
        states : torch.Tensor
            The states of the environment from the previous step.

        Returns
        -------
        action : torch.Tensor
            The action taken.
        states : torch.Tensor
            The states of the environment.
        rewards : torch.Tensor
            The rewards of the environment.
        terminated : bool
            Whether the game is terminated.

        Notes
        -----
        Something off with the `environment.step()`, as it needs to be repeated to work. Look
        more into this.
        """
        action = self.action(states)

        rewards = torch.tensor([0.0])
        states = torch.zeros(self.shape)

        for i in range(0, self.shape[2]):
            new_state, reward, terminated, truncated, _ = environment.step(action.item())     # noqa
            new_state = torch.tensor(new_state, dtype=torch.float32).view((1, 1, 210, 160))

            rewards += reward
            states[0, 0, i] = new_state

        # See note above.
        #
        # for i in range(2, self.shape[2]):
        #     new_state, reward, terminated, truncated, _ = environment.step(0)
        #     new_state = torch.tensor(new_state, dtype=torch.float32).view((1, 1, 210, 160))
        #
        #     rewards += reward
        #     states[0, 0, i] = new_state

        return action.to(self.device), states, rewards, (terminated or truncated)             # noqa

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
        memory = random.sample(self.memory["memory"],
                               min(self.memory["batch_size"], len(self.memory["memory"])))

        states = torch.cat([torch.stack(game.state).squeeze()
                            for game in memory]).unsqueeze(1)
        actions = torch.cat([torch.stack(game.action)
                             for game in memory]).to(self.device)
        new_states = torch.cat([torch.stack(game.new_states).squeeze()
                                for game in memory]).unsqueeze(1)
        rewards = torch.cat([torch.stack(game.reward)
                             for game in memory]).to(self.device)

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
        rewards = ((rewards - rewards.mean()) / (rewards.std() + 1e-7)).view(-1, 1)

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

        # Clip gradients between -1 and +1, as per Google DeepMind's suggestion.
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        # EXPLORATION RATE DECAY
        # ------------------------------------------------------------------------------------------

        self.parameter["rate"] = max(self.parameter["decay"] * self.parameter["rate"],
                                     self.parameter["min"])

        return (loss.item() / steps[-1]) * 10000

    def remember(self, *args):
        """
        Append state, action, new_state and reward along with the observerd states to agents
        memory of the
        current
        game.

        Parameters
        ----------
        *args : list of torch.Tensor
            Positional arguments to memorize.
            Must be in the following order: state, action, new_state, reward.
        """
        self.memory["game"].append(self.Game(*args))

    def memorize(self, steps):
        """
        Append game to agent memory for mini-batch training.

        Parameters
        ----------
        steps : int
            Number of steps in the game (i.e., game length).
        """
        self.memory["memory"].append(self.Memory(*zip(*self.memory["game"]), steps))
